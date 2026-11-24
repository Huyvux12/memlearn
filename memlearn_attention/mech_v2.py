"""Causal test of the source-order effect on a frozen Qwen (docs/MECH_V2_PLAN.md).

Blocks attention from one source span to another during the prompt forward pass (all layers, all heads),
with a sham block of equal token count, and scores candidates with the v2 log-prob probe.
"""
import argparse
import contextlib
import importlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
import torch
from transformers.models.qwen3 import modeling_qwen3 as native
from . import __version__
from . import data_v2,probe_v2
from .attention import _NATIVE_EAGER,_dispatch
from .runner import Store,BudgetExceeded,atomic_json,model_and_tokenizer,source_hash,sync,write_csv

# (variant, source order, blocked query span, blocked key span, layer band); band None = every layer.
VARIANTS=(('M0','book_first',None,None,None),('M1','book_first','rule','book',None),('M2','book_first','rule','sham',None),
          ('M3','rule_first',None,None,None),('M4','rule_first','book','rule',None),('M5','rule_first','book','sham',None),
          ('M6','rule_first_bookrepeat',None,None,None),
          # E2 (docs/PAPER_A_PLAN.md): narrower knockouts that may separate composition from reading the rule.
          ('N3','book_first','rule','book','early'),('N4','book_first','rule','book','mid'),('N5','book_first','rule','book','late'),
          ('N6','book_first','rule','book_keys',None),('N7','book_first','rule','book_values',None))
E2_VARIANTS=['M0','M1','M2','N3','N4','N5','N6','N7']

def band_layers(band,n_layers):
    if band is None:return list(range(n_layers))
    third=n_layers//3;start={'early':0,'mid':third,'late':2*third}[band]
    return list(range(start,n_layers if band=='late' else start+third))

def preset(name):
    # No distractor: the sham must match the rule span size, and Q would double it beyond the contract length.
    base=dict(name=name,benchmark='v2',model='Qwen/Qwen3-4B-Instruct-2507',revision='main',data_seed=17,n_test=24,n_ood=24,
              distractors=['none'],modes=['APPLY_P','NONAPPLY','READ_RULE'],variants=[v[0] for v in VARIANTS],
              dtype='float16',device='cuda',max_length=1024,budget_seconds=900,out='mech_v2',execution_kind='PRETRAINED_QWEN',
              allow_any_arch=True)
    if name=='mech_e2':base.update(variants=E2_VARIANTS,out='mech_e2')
    elif name=='mech_smoke':
        # The 2-layer test model cannot host thirds-of-depth bands; band variants are tested on a 3-layer model.
        base.update(model='RANDOM_TINY_QWEN',revision='TEST_ONLY',n_test=1,n_ood=1,dtype='float32',device='cpu',max_length=8192,
                    budget_seconds=300,execution_kind='RANDOM_QWEN_TEST_ONLY',variants=[v[0] for v in VARIANTS if v[4] is None])
    elif name!='mech_v2':raise ValueError(name)
    return base

def detect_style(text):
    """Which task format a rendered prompt uses ('codes' or one of data_v2.NL_TEMPLATES)."""
    for style in ('codes',)+tuple(data_v2.NL_TEMPLATES):
        if re.search(data_v2.span_patterns(style)['book'],text):return style
    raise ValueError('No known codebook line in prompt')

def char_spans(text):
    """Character spans of the codebook line(s), rule section and the system contract (chat-template independent)."""
    pat=data_v2.span_patterns(detect_style(text))
    book=[m.span() for m in re.finditer(pat['book'],text)]
    rule=[m.span() for m in re.finditer(pat['rule'],text)]
    framing=text.find(data_v2.DEFAULT_FRAMING)
    if framing>=0:rule.append((framing,framing+len(data_v2.DEFAULT_FRAMING)))
    keys=[];values=[]
    for s0,e0 in book:
        for m in re.finditer(pat['book_entry'],text[s0:e0]):
            keys.append((s0+m.start(1),s0+m.end(1)));values.append((s0+m.start(2),s0+m.end(2)))
    contract=next(c for c in [data_v2.CONTRACT]+[t['contract'] for t in data_v2.NL_TEMPLATES.values()] if c[:60] in text)
    start=text.index(contract[:60])
    return dict(book=book,rule=rule,book_keys=keys,book_values=values,contract=(start,start+len(contract)))

def token_spans(tok,text,n_tokens):
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);offsets=enc['offset_mapping'][:n_tokens]
    spans=char_spans(text)
    def tokens(ranges):return [i for i,(a,b) in enumerate(offsets) if any(a<e and b>s for s,e in ranges)]
    out=dict(book=tokens(spans['book']),rule=tokens(spans['rule']),book_keys=tokens(spans['book_keys']),
             book_values=tokens(spans['book_values']),ids=list(enc['input_ids'][:n_tokens]))
    contract=tokens([spans['contract']])
    # Sham: as many contract tokens as the blocked key span would have, from the contract middle; never position 0 (sink).
    def sham(n):
        mid=max(1,len(contract)//2-n//2);chosen=[i for i in contract[mid:mid+n] if i>0]
        return chosen if len(chosen)==n else None  # too long to sham-match (e.g. with distractor Q); fails only if used
    out['sham_for_book']=sham(len(out['book']));out['sham_for_rule']=sham(len(out['rule']))
    return out

_ACTIVE={'mask':None,'applied':0}

def _blocking_eager(module,query,key,value,attention_mask,scaling=None,dropout=0.0,**kwargs):
    heads=_ACTIVE.get('head_masks');band=_ACTIVE.get('layers')
    block=heads.get(module.layer_idx) if heads else _ACTIVE['mask']
    # Keywords, not positions: Gemma's kernel orders (attention_mask, dropout, scaling), Qwen/Llama's (attention_mask, scaling, dropout).
    if heads and block is None:return _ACTIVE['base'](module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
    if (block is not None and attention_mask is not None and query.shape[-2]==block.shape[-2] and key.shape[-2]==block.shape[-1]
            and (band is None or module.layer_idx in band)):
        attention_mask=attention_mask+block.to(attention_mask.dtype);_ACTIVE['applied']+=1
    return _ACTIVE['base'](module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)

# attention.attach() leaves its adapter-free-safe dispatcher installed; both it and the native kernel are safe bases.
_ORIGINAL=_NATIVE_EAGER
_SAFE_BASES=(_NATIVE_EAGER,_dispatch)

def family_module(model):
    """The HF modeling module whose eager_attention_forward this model calls (Qwen3, Qwen2, Phi3, Llama, ...)."""
    module=importlib.import_module(type(model).__module__)
    if not hasattr(module,'eager_attention_forward'):raise ValueError('Architecture has no patchable eager_attention_forward')
    return module

@contextlib.contextmanager
def blocked(q_idx,k_idx,length,device,layers=None,module=native,heads=None,n_heads=None):
    """Additive -inf from q_idx rows to k_idx columns, only for the full-prompt forward (shape length x length).

    heads: optional {layer: [head,...]}; only those heads are blocked (mask (1,H,L,L)); layers without heads are untouched."""
    previous=module.eager_attention_forward
    if previous is _blocking_eager or (module is native and previous not in _SAFE_BASES):raise RuntimeError('Another attention patch is active')
    mask=torch.zeros(1,1,length,length,device=device)
    if q_idx and k_idx:mask[0,0,torch.tensor(q_idx)[:,None],torch.tensor(k_idx)[None,:]]=float('-inf')
    head_masks=None
    if heads:
        head_masks={}
        for layer,hs in heads.items():
            m=torch.zeros(1,n_heads,length,length,device=device)
            for h in hs:m[0,h]=mask[0,0]
            head_masks[layer]=m
        layers=list(heads)
    _ACTIVE.update(mask=mask,applied=0,base=previous,layers=None if layers is None else set(layers),head_masks=head_masks);module.eager_attention_forward=_blocking_eager
    try:yield _ACTIVE
    finally:_ACTIVE.update(mask=None,head_masks=None);module.eager_attention_forward=previous

def score_variant(model,tok,item,variant,device,max_length):
    _,_,qspan,kspan,band=variant
    if qspan is None:return probe_v2.score_candidates(model,tok,item,device,max_length),0
    cands,seqs,start=probe_v2.sequences(tok,item)
    # The shared prefix can extend into the candidates (e.g. the common 'z'), so map offsets on a full candidate string.
    text=probe_v2.prompt_text(tok,item)+data_v2.prefix(item['bundle'],item['mode'])+cands[0][1]+'"}'
    spans=token_spans(tok,text,start)
    if spans['ids']!=seqs[0][:start]:raise ValueError('Offset tokenization differs from scored prefix')
    # Sham keys match the size of the real key span: rule queries -> book-sized sham; book queries -> rule-sized sham.
    keys=(spans['sham_for_book'] if qspan=='rule' else spans['sham_for_rule']) if kspan=='sham' else spans[kspan]
    if keys is None:raise ValueError('Contract too short for an equal-size sham span')
    queries=spans[qspan]
    if not queries or not keys:raise ValueError('Empty span')
    layers=band_layers(band,len(model.model.layers))
    if not layers:raise ValueError(f'Layer band {band} is empty for {len(model.model.layers)} layers')
    with blocked(queries,keys,start,device,None if band is None else layers,family_module(model)) as state:
        scored=probe_v2.score_candidates(model,tok,item,device,max_length)
        # The block must reach every targeted decoder layer of the prompt forward, or the intervention silently did nothing.
        if state['applied']!=len(layers):raise RuntimeError(f"Block applied in {state['applied']} of {len(layers)} layers")
    return scored,len(queries)*len(keys)

def run(cfg):
    cfg=dict(cfg);store=Store(cfg['out'],cfg);start=time.monotonic();deadline=start+cfg['budget_seconds'];status='COMPLETE';error=None
    variants=[v for v in VARIANTS if v[0] in cfg['variants']]
    ds=data_v2.dataset(cfg['n_test'],cfg['n_ood'],seed=cfg['data_seed'],style=cfg.get('style','codes'));atomic_json(store.root/'dataset_v2.json',ds)
    work=[]
    for split,bundles in ds.items():
        for b in bundles:
            for d in cfg['distractors']:
                for mode in cfg['modes']:
                    for v in variants:
                        item=dict(bundle=b,bundle_id=b['id'],split=split,family=b['family'],mode=mode,condition=('neutral',d,v[1]),control='FULL')
                        item['id']=data_v2.digest([b['id'],mode,d,v[0]])[:24];work.append((item,v))
    try:
        model,tok,identity=model_and_tokenizer(cfg,store)
        atomic_json(store.root/'manifest.json',dict(version=__version__,source_sha256=source_hash(),config_sha256=store.fingerprint,model=identity,
                                                    torch=torch.__version__,python=sys.version,benchmark='v2-mechanism',
                                                    intervention='additive -inf attention block, all layers and heads, prompt forward only',
                                                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        done={store.key(dict(id=r['id'],method=r['variant'],seed=0)) for r in store.load('mech')}
        for item,v in work:
            key=dict(id=item['id'],method=v[0],seed=0)
            if store.key(key) in done:continue
            if time.monotonic()>deadline:raise BudgetExceeded('Mechanism budget exhausted')
            sync(cfg['device']);t0=time.monotonic()
            scored,blocked_pairs=score_variant(model,tok,item,v,cfg['device'],cfg['max_length']);sync(cfg['device'])
            row=dict(key,variant=v[0],control=v[0],source_order=v[1],blocked_query=v[2],blocked_key=v[3],layer_band=v[4],blocked_pairs=blocked_pairs,
                     bundle_id=item['bundle_id'],split=item['split'],family=item['family'],mode=item['mode'],condition=list(item['condition']),
                     latency_s=time.monotonic()-t0,**scored,**probe_v2.decide(item,scored))
            store.save('mech',row);done.add(store.key(key))
            if len(done)%100==0:print(f'MECH {len(done)}/{len(work)} elapsed={time.monotonic()-start:.0f}s',flush=True)
    except BudgetExceeded as e:status='BUDGET_EXHAUSTED';error=str(e)
    except Exception as e:status='RUNTIME_ERROR';error=repr(e);(store.root/'ERROR.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        rows=store.load('mech')
        table={}
        for r in rows:table.setdefault((r['variant'],r['mode']),[]).append(r)
        write_csv(store.root/'mech_by_variant.csv',[dict(variant=k[0],mode=k[1],n=len(v),accuracy=sum(x['correct'] for x in v)/len(v),
                                                         leakage=sum(x['leakage'] for x in v)/len(v),
                                                         stop_at_base=sum(x['choice']=='base' for x in v)/len(v),
                                                         mean_p_gold=sum(x['p_gold'] for x in v)/len(v)) for k,v in sorted(table.items())])
        atomic_json(store.root/'run_status.json',dict(status=status,error=error,completed=len(rows),expected=len(work),elapsed_s=time.monotonic()-start))
    return dict(status=status,error=error,completed=len(rows),expected=len(work))

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['mech_smoke','mech_v2','mech_e2'],default='mech_v2')
    p.add_argument('--out',default='mech_v2');p.add_argument('--config');a=p.parse_args()
    cfg=preset(a.preset)
    if a.config:cfg.update(json.loads(Path(a.config).read_text()))
    cfg['out']=str(Path(a.out).resolve());result=run(cfg);print(json.dumps(result,indent=2))
    if result['status']=='RUNTIME_ERROR':sys.exit(2)

if __name__=='__main__':main()
