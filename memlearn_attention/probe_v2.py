"""Format-free decision probe for benchmark v2 on a frozen Qwen (docs/BENCHMARK_V2_SPEC.md).

Scores fixed candidate answers by log-probability after a teacher-forced JSON prefix.
No generation, no gold in the prompt, no adapters in this version.
"""
import argparse
import copy
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
import torch
from . import __version__
from . import data_v2
from .runner import Store,BudgetExceeded,atomic_json,model_and_tokenizer,source_hash,sync,write_csv

def preset(name):
    base=dict(name=name,benchmark='v2',model='Qwen/Qwen3-4B-Instruct-2507',revision='main',data_seed=17,n_test=24,n_ood=24,
              controls=['FULL'],dtype='float16',device='cuda',max_length=1024,budget_seconds=900,out='probe_v2',
              execution_kind='PRETRAINED_QWEN',allow_any_arch=True)
    if name=='probe_smoke':
        base.update(model='RANDOM_TINY_QWEN',revision='TEST_ONLY',n_test=1,n_ood=1,controls=['FULL','SCOPE_PROMPT'],
                    dtype='float32',device='cpu',max_length=4096,budget_seconds=300,execution_kind='RANDOM_QWEN_TEST_ONLY')
    elif name!='probe_v2':raise ValueError(name)
    return base

# Llama-3.2's template writes today's date into the system turn; pin it (Llama-3.1's own default) so prompts never depend on the run date.
TEMPLATE_KW=dict(tokenize=False,add_generation_prompt=True,enable_thinking=False,date_string='26 Jul 2024')

def prompt_text(tok,item):
    msgs=data_v2.messages(item)
    try:return tok.apply_chat_template(msgs,**TEMPLATE_KW)
    except Exception as e:
        # Templates without a system role (Gemma) raise; the contract then opens the user turn. Other models never reach this.
        if 'system' not in str(e).lower():raise
        merged=[{'role':'user','content':msgs[0]['content']+'\n\n'+msgs[1]['content']}]
        return tok.apply_chat_template(merged,**TEMPLATE_KW)

def sequences(tok,item):
    b=item['bundle'];head=prompt_text(tok,item)+data_v2.prefix(b,item['mode'])
    cands=data_v2.candidates(b,item['mode'])
    seqs=[tok.encode(head+value+'"}',add_special_tokens=False) for _,value in cands]
    # Score from the longest token prefix shared by ALL candidates, so boundary merges cannot favour one.
    start=0
    while all(len(s)>start for s in seqs) and len({s[start] for s in seqs})==1:start+=1
    if start==0:raise ValueError('Candidates share no prompt prefix')
    if any(len(s)==start for s in seqs):raise ValueError('A candidate is a token prefix of another')
    return cands,seqs,start

def score_candidates(model,tok,item,device,max_length):
    cands,seqs,start=sequences(tok,item)
    if max(len(s) for s in seqs)>max_length:raise ValueError(f'Overlong probe sequence > {max_length}')
    with torch.no_grad():
        out=model(input_ids=torch.tensor([seqs[0][:start]],device=device),use_cache=True,logits_to_keep=1)
        first=out.logits[0,-1].float().log_softmax(-1);cache=out.past_key_values
        scores=[float(first[s[start]]) for s in seqs]
        tails=[s[start:-1] for s in seqs]
        if len({len(t) for t in tails})==1 and len(tails[0])>0:
            # Equal-length tails (the usual case): one batched forward over a repeated prefix cache.
            cache.batch_repeat_interleave(len(seqs))
            logp=model(input_ids=torch.tensor(tails,device=device),past_key_values=cache,use_cache=True).logits.float().log_softmax(-1)
            for row,s in enumerate(seqs):scores[row]+=float(sum(logp[row,i,t] for i,t in enumerate(s[start+1:])))
        else:
            for row,s in enumerate(seqs):
                if len(s)>start+1:
                    tail=model(input_ids=torch.tensor([s[start:-1]],device=device),past_key_values=copy.deepcopy(cache),use_cache=True).logits[0].float().log_softmax(-1)
                    scores[row]+=float(sum(tail[i,t] for i,t in enumerate(s[start+1:])))
    return dict(labels=[l for l,_ in cands],values=[v for _,v in cands],logprobs=scores,scored_from_token=start,prompt_tokens=start)

FIELD={'READ_DEC':'decoded_value','READ_RULE':'rule_result'}

def generate_choice(model,tok,item,device,max_new_tokens=32):
    """Greedy free generation (no teacher forcing); maps the answer field onto a candidate label or 'invalid'."""
    ids=tok.encode(prompt_text(tok,item),add_special_tokens=False)
    with torch.no_grad():
        out=model.generate(torch.tensor([ids],device=device),attention_mask=torch.ones(1,len(ids),dtype=torch.long,device=device),
                           max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    text=tok.decode(out[0,len(ids):].tolist(),skip_special_tokens=True)
    value=None
    try:
        start=text.index('{');obj=json.loads(text[start:text.index('}',start)+1])
        value=obj.get(FIELD.get(item['mode'],'answer'))
    except (ValueError,AttributeError):pass
    labels={v:l for l,v in data_v2.candidates(item['bundle'],item['mode'])}
    return dict(gen_text=text[:200],gen_choice=labels.get(value,'invalid') if isinstance(value,str) else 'invalid')

def decide(item,scored):
    labels=scored['labels'];lp=scored['logprobs'];best=labels[max(range(len(lp)),key=lp.__getitem__)]
    z=max(lp);probs=[math.exp(x-z) for x in lp];total=sum(probs);probs=[x/total for x in probs]
    by=dict(zip(labels,lp));mode=item['mode']
    return dict(choice=best,correct=int(best=='gold'),leakage=int(mode=='NONAPPLY' and best=='p'),
                cross_rule=int(mode=='APPLY_P' and best=='q'),p_gold=probs[labels.index('gold')],
                margin_gold_vs_p=by['gold']-by['p'] if 'p' in by else None)

def summarize(rows):
    cells=defaultdict(list)
    for r in rows:
        f,d,o=r['condition']
        for key in ((r['control'],r['split'],f,d,o,r['mode']),(r['control'],r['split'],f,d,'pooled',r['mode'])):cells[key].append(r)
    table=[]
    for (control,split,f,d,o,mode),rs in sorted(cells.items()):
        margins=[r['margin_gold_vs_p'] for r in rs if r['margin_gold_vs_p'] is not None]
        table.append(dict(control=control,split=split,framing=f,distractor=d,source_order=o,mode=mode,n=len(rs),
                          accuracy=sum(r['correct'] for r in rs)/len(rs),leakage=sum(r['leakage'] for r in rs)/len(rs),
                          cross_rule=sum(r['cross_rule'] for r in rs)/len(rs),mean_p_gold=sum(r['p_gold'] for r in rs)/len(rs),
                          mean_margin_gold_vs_p=sum(margins)/len(margins) if margins else None))
    joint=defaultdict(dict)
    for r in rows:joint[(r['control'],r['split'],tuple(r['condition']),r['bundle_id'])][r['mode']]=r['correct']
    jss=defaultdict(list)
    for (control,split,cond,_),modes in joint.items():
        if all(m in modes for m in ('READ_DEC','APPLY_P','NONAPPLY')):
            jss[(control,split,cond[0],cond[1])].append(int(modes['READ_DEC'] and modes['APPLY_P'] and modes['NONAPPLY']))
    jss_rows=[dict(control=c,split=s,framing=f,distractor=d,n=len(v),jss_v2=sum(v)/len(v)) for (c,s,f,d),v in sorted(jss.items())]
    return table,jss_rows

def target_failure(table,min_n=32,leak=.10,apply_acc=.70):
    """Pre-registered rule from the spec: FULL NONAPPLY leakage >= 10% where APPLY_P accuracy >= 70%, pooled over order."""
    pooled={(r['split'],r['framing'],r['distractor'],r['mode']):r for r in table if r['control']=='FULL' and r['source_order']=='pooled'}
    cells=[]
    for (split,f,d,mode),r in pooled.items():
        if mode!='NONAPPLY':continue
        a=pooled.get((split,f,d,'APPLY_P'))
        cells.append(dict(split=split,framing=f,distractor=d,n=r['n'],leakage=r['leakage'],apply_p_accuracy=a['accuracy'] if a else None,
                          meets=bool(r['n']>=min_n and r['leakage']>=leak and a and a['accuracy']>=apply_acc)))
    return dict(rule=f'FULL NONAPPLY leakage >= {leak} with APPLY_P accuracy >= {apply_acc}, n >= {min_n} per pooled cell',
                target_failure_exists=any(c['meets'] for c in cells),cells=cells)

def run(cfg):
    cfg=dict(cfg);store=Store(cfg['out'],cfg);start=time.monotonic();deadline=start+cfg['budget_seconds'];status='COMPLETE';error=None
    ds=data_v2.dataset(cfg['n_test'],cfg['n_ood'],seed=cfg['data_seed'],style=cfg.get('style','codes'));atomic_json(store.root/'dataset_v2.json',ds)
    conditions=[tuple(c) for c in cfg['conditions']] if cfg.get('conditions') else data_v2.CONDITIONS
    all_items=data_v2.items(ds,tuple(cfg['controls']),conditions,tuple(cfg.get('modes',data_v2.MODES)))
    # Bundle-major order: a budget stop leaves complete, balanced bundles rather than whole missing conditions.
    order={b['id']:i for split in ds.values() for i,b in enumerate(split)}
    all_items.sort(key=lambda x:(order[x['bundle_id']],x['split']))
    try:
        model,tok,identity=model_and_tokenizer(cfg,store)
        atomic_json(store.root/'manifest.json',dict(version=__version__,source_sha256=source_hash(),config_sha256=store.fingerprint,model=identity,
                                                    torch=torch.__version__,python=sys.version,benchmark='v2',evaluation='teacher-forced candidate log-probabilities',
                                                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        done={store.key(dict(id=r['id'],method=r['control'],seed=0)) for r in store.load('probe')}
        for item in all_items:
            key=dict(id=item['id'],method=item['control'],seed=0)
            if store.key(key) in done:continue
            if time.monotonic()>deadline:raise BudgetExceeded('Probe budget exhausted')
            sync(cfg['device']);t0=time.monotonic()
            scored=score_candidates(model,tok,item,cfg['device'],cfg['max_length']);sync(cfg['device'])
            row=dict(key,control=item['control'],bundle_id=item['bundle_id'],split=item['split'],family=item['family'],mode=item['mode'],
                     condition=list(item['condition']),latency_s=time.monotonic()-t0,**scored,**decide(item,scored))
            # Optional free-generation agreement check on a deterministic subset (every k-th item by id hash).
            k=cfg.get('generate_every',0)
            if k and int(item['id'][:8],16)%k==0:
                row.update(generate_choice(model,tok,item,cfg['device']));row['gen_agree']=int(row['gen_choice']==row['choice'])
            store.save('probe',row);done.add(store.key(key))
            if len(done)%50==0:print(f'PROBE {len(done)}/{len(all_items)} elapsed={time.monotonic()-start:.0f}s',flush=True)
    except BudgetExceeded as e:status='BUDGET_EXHAUSTED';error=str(e)
    except Exception as e:status='RUNTIME_ERROR';error=repr(e);(store.root/'ERROR.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        rows=store.load('probe');table,jss=summarize(rows) if rows else ([],[])
        write_csv(store.root/'probe_by_cell.csv',table);write_csv(store.root/'probe_jss_v2.csv',jss)
        verdict=target_failure(table) if rows else None
        atomic_json(store.root/'probe_decision.json',verdict)
        atomic_json(store.root/'run_status.json',dict(status=status,error=error,completed=len(rows),expected=len(all_items),elapsed_s=time.monotonic()-start))
    return dict(status=status,error=error,completed=len(rows),expected=len(all_items),decision=verdict)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['probe_smoke','probe_v2'],default='probe_v2')
    p.add_argument('--out',default='probe_v2');p.add_argument('--config');a=p.parse_args()
    cfg=preset(a.preset)
    if a.config:cfg.update(json.loads(Path(a.config).read_text()))
    cfg['out']=str(Path(a.out).resolve());result=run(cfg);print(json.dumps(result,indent=2))
    if result['status']=='RUNTIME_ERROR':sys.exit(2)

if __name__=='__main__':main()
