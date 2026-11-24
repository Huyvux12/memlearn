"""Sufficiency, head localization and a two-stage circuit for the cross-source binding (docs/E4_PLAN in ROUND3/ROUND4_PLAN).

Terminology: the *later* source is the one placed second (the rule in codebook-first order, the codebook in rule-first).
(a) Restoration: block later -> earlier attention in every layer, restore clean later-source states after layer L.
(b) Binding heads: attention from a later-source value token to the same value in the earlier source (duplicate-token score).
(c) Binding-head necessity (top{k}/rand{k}_r) and sufficiency (only_top{k}: every other head is blocked).
(d) Reader heads: attention from the answer position to the queried rule entry; necessity (read_top{k}/read_rand{k}_r).
(e) Path: reader attention to the queried entry when binding heads (top vs random) are blocked.
Scored with the v2 log-prob probe; neutral framing, no distractor.
"""
import argparse
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
import torch
from . import __version__
from . import data_v2,probe_v2,mech_v2
from .runner import Store,BudgetExceeded,atomic_json,model_and_tokenizer,source_hash,sync,write_csv

def preset(name):
    base=dict(name=name,benchmark='v2-e4',model='Qwen/Qwen3-4B-Instruct-2507',revision='main',data_seed=17,n_test=24,n_ood=24,
              style='codes',order='book_first',modes=['APPLY_P','READ_RULE'],restore_every=2,ks=[8,32],suff_ks=[32],read_ks=[8],
              random_draws=3,head_depth=2/3,score_items=32,
              dtype='float16',device='cuda',max_length=1024,budget_seconds=1500,out='mech_e4',execution_kind='PRETRAINED_QWEN',allow_any_arch=True)
    if name=='e4_smoke':
        base.update(model='RANDOM_TINY_QWEN',revision='TEST_ONLY',n_test=1,n_ood=1,restore_every=1,ks=[1],suff_ks=[1],read_ks=[1],
                    random_draws=2,head_depth=1.,score_items=2,dtype='float32',device='cpu',max_length=8192,budget_seconds=600,
                    execution_kind='RANDOM_QWEN_TEST_ONLY')
    elif name!='mech_e4':raise ValueError(name)
    return base

def make_items(cfg):
    ds=data_v2.dataset(cfg['n_test'],cfg['n_ood'],seed=cfg['data_seed'],style=cfg.get('style','codes'))
    cond=('neutral','none',cfg.get('order','book_first'))
    return [dict(bundle=b,bundle_id=b['id'],split=s,family=b['family'],mode=m,condition=cond,control='FULL',
                 # v2.0.0 ids for codebook-first; the order joins the id otherwise.
                 id=data_v2.digest([b['id'],m,'e4']+([] if cond[2]=='book_first' else [cond[2]]))[:24]) for s,bs in ds.items() for b in bs for m in cfg['modes']]

def layout(tok,item):
    """Token positions of both sources, per-value tokens in each, and each rule entry (for reader heads)."""
    cands,seqs,start=probe_v2.sequences(tok,item)
    text=probe_v2.prompt_text(tok,item)+data_v2.prefix(item['bundle'],item['mode'])+cands[0][1]+'"}'
    spans=mech_v2.token_spans(tok,text,start)
    if spans['ids']!=seqs[0][:start]:raise ValueError('Offset tokenization differs from scored prefix')
    offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'][:start]
    def toks(a,b):return [i for i,(x,y) in enumerate(offsets) if x<b and y>a]
    pat=data_v2.span_patterns(item['bundle'].get('style','codes'))
    book=re.search(pat['book'],text);rule=re.search(pat['rule_p'],text)
    book_vals={m.group(2):toks(book.start()+m.start(2),book.start()+m.end(2)) for m in re.finditer(pat['book_entry'],book.group(0))}
    rule_left={m.group(1):toks(rule.start()+m.start(1),rule.start()+m.end(1)) for m in re.finditer(pat['rule_entry'],rule.group(0))}
    rule_entry={m.group(1):toks(rule.start()+m.start(0),rule.start()+m.end(0)) for m in re.finditer(pat['rule_entry'],rule.group(0))}
    bf=item['condition'][2]=='book_first'
    return dict(start=start,rule=spans['rule'],book=spans['book'],book_values=spans['book_values'],book_vals=book_vals,rule_left=rule_left,
                rule_entry=rule_entry,base=data_v2.facts(item['bundle'],item['mode'])['base'],
                later=spans['rule'] if bf else spans['book'],earlier=spans['book'] if bf else spans['rule'],
                later_vals=rule_left if bf else book_vals,earlier_vals=book_vals if bf else rule_left,
                earlier_value_tokens=spans['book_values'] if bf else sorted(i for v in rule_left.values() for i in v))

def clean_states(model,ids,positions):
    """Per-layer outputs at `positions`, captured with forward hooks (independent of hidden_states indexing)."""
    saved={};hooks=[]
    for i,layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(lambda mod,inp,out,i=i:saved.__setitem__(i,(out[0] if isinstance(out,tuple) else out)[:,positions].detach().clone())))
    try:
        with torch.no_grad():model(input_ids=ids,use_cache=False)
    finally:
        for h in hooks:h.remove()
    return saved

clean_rule_states=clean_states  # v2.0.0 name

def restore_hook(layer,positions,states,length):
    def hook(mod,inp,out):
        h=out[0] if isinstance(out,tuple) else out
        if h.shape[1]!=length:return None  # only the prompt pass, never the candidate tails
        h=h.clone();h[:,positions]=states.to(h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
    return layer.register_forward_hook(hook)

def _attentions(model,ids):
    with torch.no_grad():att=model(input_ids=ids,output_attentions=True,use_cache=False).attentions
    if att is None or len(att)!=len(model.model.layers):raise RuntimeError('Model did not return per-layer attentions')
    return torch.stack([a[0].float() for a in att])  # layers x heads x S x S

def _ids(tok,item,lay,device):return torch.tensor([probe_v2.sequences(tok,item)[1][0][:lay['start']]],device=device)

def binding_scores(model,tok,items,device):
    """Mean over items/values of attention(later v -> earlier v) minus mean attention(later v -> earlier u != v)."""
    total=None;count=0
    for item in items:
        lay=layout(tok,item);A=_attentions(model,_ids(tok,item,lay,device))
        for v,qpos in lay['later_vals'].items():
            if v not in lay['earlier_vals'] or not qpos:continue
            sub=A[:,:,qpos]
            match=sub[:,:,:,lay['earlier_vals'][v]].sum(-1).mean(-1)
            others=[sub[:,:,:,pos].sum(-1).mean(-1) for u,pos in lay['earlier_vals'].items() if u!=v]
            score=match-torch.stack(others).mean(0);total=score if total is None else total+score;count+=1
    return (total/count).cpu()

head_scores=binding_scores  # v2.0.0 name

def reader_mass(A,lay):
    """Attention from the answer position (start-1) to the queried rule entry, minus the mean over the other entries."""
    q=lay['start']-1;ent=lay['rule_entry']
    if lay['base'] not in ent:return None
    match=A[:,:,q,ent[lay['base']]].sum(-1)
    others=torch.stack([A[:,:,q,pos].sum(-1) for v,pos in ent.items() if v!=lay['base']]).mean(0)
    return match-others

def reader_scores(model,tok,items,device):
    total=None;count=0
    for item in items:
        lay=layout(tok,item);s=reader_mass(_attentions(model,_ids(tok,item,lay,device)),lay)
        if s is None:continue
        total=s if total is None else total+s;count+=1
    return (total/count).cpu()

def _heads(pairs):
    out={}
    for l,h in pairs:out.setdefault(l,[]).append(h)
    return out

def run(cfg):
    cfg=dict(cfg);store=Store(cfg['out'],cfg);start=time.monotonic();deadline=start+cfg['budget_seconds'];status='COMPLETE';error=None
    items=make_items(cfg);atomic_json(store.root/'items.json',[{k:v for k,v in x.items() if k!='bundle'} for x in items])
    expected=0
    try:
        model,tok,identity=model_and_tokenizer(cfg,store);family=mech_v2.family_module(model)
        n_layers=len(model.model.layers);n_heads=model.config.num_attention_heads
        restore_layers=list(range(0,n_layers-1,cfg['restore_every']))
        atomic_json(store.root/'manifest.json',dict(version=__version__,source_sha256=source_hash(),config_sha256=store.fingerprint,model=identity,
                                                    n_layers=n_layers,n_heads=n_heads,restore_layers=restore_layers,torch=torch.__version__,
                                                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        scoring=items[:cfg['score_items']]
        bind=binding_scores(model,tok,scoring,cfg['device']);read=reader_scores(model,tok,scoring,cfg['device'])
        depth=max(1,int(round(n_layers*cfg['head_depth'])))
        pool=[(l,h) for l in range(depth) for h in range(n_heads)];every=[(l,h) for l in range(n_layers) for h in range(n_heads)]
        ranked=sorted(pool,key=lambda lh:-float(bind[lh[0],lh[1]]));rranked=sorted(every,key=lambda lh:-float(read[lh[0],lh[1]]))
        rng=random.Random(cfg['data_seed']);sets={}
        for k in cfg['ks']:
            sets[f'top{k}']=('bind',ranked[:k])
            for r in range(cfg['random_draws']):sets[f'rand{k}_{r}']=('bind',rng.sample(pool,k))
        for k in cfg['suff_ks']:sets[f'only_top{k}']=('bind_all_but',ranked[:k])
        for k in cfg['read_ks']:
            sets[f'read_top{k}']=('read',rranked[:k])
            for r in range(cfg['random_draws']):sets[f'read_rand{k}_{r}']=('read',rng.sample(every,k))
        atomic_json(store.root/'head_scores.json',dict(scores=bind.tolist(),reader_scores=read.tolist(),depth=depth,top=[list(x) for x in ranked[:64]],
                                                       reader_top=[list(x) for x in rranked[:64]],head_sets={k:[list(x) for x in v[1]] for k,v in sets.items()},
                                                       kinds={k:v[0] for k,v in sets.items()}))
        variants=['clean','blocked']+[f'restore_L{l}' for l in restore_layers]+list(sets)
        expected=len(variants)*len(items)
        done={store.key(dict(id=r['id'],method=r['variant'],seed=0)) for r in store.load('e4')}
        path_done={store.key(dict(id=r['id'],method=r['variant'],seed=0)) for r in store.load('e4path')}
        k_path=cfg['read_ks'][0];readers=rranked[:k_path];k0=cfg['ks'][0]
        path_variants=['clean',f'top{k0}']+[f'rand{k0}_{r}' for r in range(cfg['random_draws'])]
        for item in items:
            lay=layout(tok,item);ids=_ids(tok,item,lay,cfg['device']);clean=None
            for variant in variants:
                key=dict(id=item['id'],method=variant,seed=0)
                if store.key(key) in done:continue
                if time.monotonic()>deadline:raise BudgetExceeded('E4 budget exhausted')
                sync(cfg['device']);t0=time.monotonic();extra={}
                if variant=='clean':scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length'])
                elif variant=='blocked' or variant.startswith('restore_L'):
                    hook=None
                    if variant.startswith('restore_L'):
                        layer=int(variant[9:])
                        if clean is None:clean=clean_states(model,ids,lay['later'])
                        hook=restore_hook(model.model.layers[layer],lay['later'],clean[layer],lay['start']);extra['restore_layer']=layer
                    try:
                        with mech_v2.blocked(lay['later'],lay['earlier'],lay['start'],cfg['device'],module=family) as state:
                            scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length'])
                            if state['applied']!=n_layers:raise RuntimeError('Block missed layers')
                    finally:
                        if hook is not None:hook.remove()
                else:
                    kind,pairs=sets[variant]
                    if kind=='bind':q,k_,heads=lay['later'],lay['earlier_value_tokens'],_heads(pairs)
                    elif kind=='bind_all_but':
                        keep=set(pairs);q,k_,heads=lay['later'],lay['earlier'],_heads([x for x in every if x not in keep])
                    else:q,k_,heads=[lay['start']-1],lay['rule'],_heads(pairs)
                    with mech_v2.blocked(q,k_,lay['start'],cfg['device'],module=family,heads=heads,n_heads=n_heads) as state:
                        scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length'])
                        if state['applied']!=len(heads):raise RuntimeError('Head block missed layers')
                    extra.update(n_heads_blocked=sum(len(v) for v in heads.values()),head_kind=kind)
                row=dict(key,variant=variant,bundle_id=item['bundle_id'],split=item['split'],mode=item['mode'],condition=list(item['condition']),
                         latency_s=time.monotonic()-t0,**extra,**scored,**probe_v2.decide(item,scored))
                store.save('e4',row);done.add(store.key(row))
            # (e) path: reader attention to the queried entry under binding-head blocks.
            for variant in path_variants:
                key=dict(id=item['id'],method=variant,seed=0)
                if store.key(key) in path_done:continue
                if variant=='clean':A=_attentions(model,ids)
                else:
                    with mech_v2.blocked(lay['later'],lay['earlier_value_tokens'],lay['start'],cfg['device'],module=family,heads=_heads(sets[variant][1]),n_heads=n_heads):
                        A=_attentions(model,ids)
                mass=reader_mass(A,lay)
                if mass is None:continue
                value=float(torch.stack([mass[l,h] for l,h in readers]).mean())
                store.save('e4path',dict(key,variant=variant,bundle_id=item['bundle_id'],split=item['split'],mode=item['mode'],reader_mass=value));path_done.add(store.key(key))
            print(f'E4 {len(done)}/{expected} elapsed={time.monotonic()-start:.0f}s',flush=True)
    except BudgetExceeded as e:status='BUDGET_EXHAUSTED';error=str(e)
    except Exception as e:status='RUNTIME_ERROR';error=repr(e);(store.root/'ERROR.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        rows=store.load('e4');table={}
        for r in rows:table.setdefault((r['variant'],r['mode']),[]).append(r['correct'])
        write_csv(store.root/'e4_by_variant.csv',[dict(variant=k[0],mode=k[1],n=len(v),accuracy=sum(v)/len(v)) for k,v in sorted(table.items())])
        atomic_json(store.root/'run_status.json',dict(status=status,error=error,completed=len(rows),expected=expected,elapsed_s=time.monotonic()-start))
    return dict(status=status,error=error,completed=len(rows),expected=expected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['e4_smoke','mech_e4'],default='mech_e4')
    p.add_argument('--out',default='mech_e4');p.add_argument('--config');a=p.parse_args()
    cfg=preset(a.preset)
    if a.config:cfg.update(json.loads(Path(a.config).read_text()))
    cfg['out']=str(Path(a.out).resolve());result=run(cfg);print(json.dumps(result,indent=2))
    if result['status']=='RUNTIME_ERROR':sys.exit(2)

if __name__=='__main__':main()
