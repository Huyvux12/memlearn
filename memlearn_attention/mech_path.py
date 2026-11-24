"""Round 5 (docs/ROUND5_PLAN.md): where the cross-source binding is written, and how the reader heads use it.

(A) Sender side. Block later -> earlier attention in every layer, then add back, at the later-source positions of the residual
    stream after layer L* (path_layer), the clean-minus-blocked direct contribution of chosen components of layers 0..L*:
    attention heads (o_proj input split by head, through its o_proj slice) and/or MLP outputs. 'dec_all' rebuilds the clean
    residual exactly, which holds for pre-norm decoders only (no norm between a head's output and the residual).
(B) Receiver side. In the blocked run, the key/value groups of the top-k reader heads get their clean keys and/or values at the
    later-source positions (k_proj/v_proj outputs, or the K/V slices of a fused qkv_proj); grouped-query models restore whole groups.
(C) Route. Full block (whole earlier source) vs value-token block (all heads) vs top-k binding-head block, scored and with the
    top reader heads' attention to the queried rule entry.
(D) Heads (round 6). Clean keys at the later-source positions for single query heads only (top-k reader heads, their group-mates,
    random heads), separating a reader head from the other query heads that share its key/value group.
`parts` selects which of sender/receiver/route/heads run; clean and blocked always run, and every head set is drawn regardless.
Heads are ranked on held-out bundles (head_seed), never on the evaluated ones. Scored with the v2 log-prob probe.
"""
import argparse
import contextlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
import torch
from . import __version__
from . import probe_v2,mech_v2,mech_e4
from .runner import Store,BudgetExceeded,atomic_json,model_and_tokenizer,source_hash,sync,write_csv

PRE_NORM=('qwen3','qwen2','phi3','mistral','llama')

def preset(name):
    base=dict(name=name,benchmark='v2-path',model='Qwen/Qwen3-4B-Instruct-2507',revision='main',data_seed=41,head_seed=53,n_test=24,n_ood=24,
              style='codes',order='book_first',modes=['APPLY_P','READ_RULE'],path_layer=16,dec_ks=[8,32,128],dec_rand_ks=[32,128],
              bind_k=32,read_k=8,random_draws=3,head_depth=2/3,score_items=32,parts=['sender','receiver','route'],head_ks=[1,2,4,8,16],
              dtype='float16',device='cuda',max_length=1024,budget_seconds=2400,out='mech_path',execution_kind='PRETRAINED_QWEN',allow_any_arch=True)
    if name=='path_smoke':
        base.update(model='RANDOM_TINY_QWEN',revision='TEST_ONLY',n_test=1,n_ood=1,path_layer=0,dec_ks=[1],dec_rand_ks=[1],bind_k=1,read_k=1,
                    parts=['sender','receiver','route','heads'],head_ks=[1,2],
                    random_draws=1,head_depth=1.,score_items=2,dtype='float32',device='cpu',max_length=8192,budget_seconds=600,
                    execution_kind='RANDOM_QWEN_TEST_ONLY')
    elif name!='mech_path':raise ValueError(name)
    return base

def _run(model,ids):
    with torch.no_grad():model(input_ids=ids,use_cache=False)

def components(model,ids,positions,upto):
    """o_proj inputs (all heads' outputs, concatenated) and MLP outputs at `positions`, layers 0..upto, in float32."""
    z={};m={};hooks=[]
    for i,layer in enumerate(model.model.layers[:upto+1]):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda mod,inp,i=i:z.__setitem__(i,inp[0][0,positions].detach().float())))
        hooks.append(layer.mlp.register_forward_hook(lambda mod,inp,out,i=i:m.__setitem__(i,out[0,positions].detach().float())))
    try:_run(model,ids)
    finally:
        for h in hooks:h.remove()
    return z,m

def delta(model,clean,blocked,heads=None,mlp=False):
    """Clean-minus-blocked direct contribution to the residual of the chosen heads ({layer:[head]}, or 'all') and, optionally, the MLPs."""
    (zc,mc),(zb,mb)=clean,blocked;n_heads=model.config.num_attention_heads;total=None
    for l in zc:
        hs=list(range(n_heads)) if heads=='all' else (heads or {}).get(l,[]);parts=[]
        if hs:
            d=zc[l]-zb[l];D=d.shape[-1]//n_heads;mask=torch.zeros(d.shape[-1],device=d.device)
            for h in hs:mask[h*D:(h+1)*D]=1
            parts.append((d*mask)@model.model.layers[l].self_attn.o_proj.weight.float().T)
        if mlp:parts.append(mc[l]-mb[l])
        for p in parts:total=p if total is None else total+p
    return total

def add_hook(layer,positions,add,length):
    def hook(mod,inp,out):
        h=out[0] if isinstance(out,tuple) else out
        if h.shape[1]!=length:return None  # only the prompt pass, never the candidate tails
        h=h.clone();h[:,positions]=(h[:,positions].float()+add).to(h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
    return layer.register_forward_hook(hook)

def kv_groups(model,pairs):
    per=model.config.num_attention_heads//model.config.num_key_value_heads;out={}
    for l,h in pairs:out.setdefault(l,set()).add(h//per)
    return {l:sorted(g) for l,g in sorted(out.items())}

def kv_targets(model,groups,which):
    """(projection module, output channels) carrying the keys ('k') and/or values ('v') of {layer:[kv group]}."""
    c=model.config;H=c.num_attention_heads;KV=c.num_key_value_heads;out=[]
    for l,gs in groups.items():
        attn=model.model.layers[l].self_attn;D=attn.head_dim
        for w in which:
            if hasattr(attn,'qkv_proj'):mod=attn.qkv_proj;base=H*D+(0 if w=='k' else KV*D)
            else:mod=getattr(attn,f'{w}_proj');base=0
            out.append((mod,torch.tensor([base+g*D+j for g in gs for j in range(D)])))
    return out

def capture(model,ids,targets,positions):
    saved={};hooks=[]
    for n,(mod,idx) in enumerate(targets):
        hooks.append(mod.register_forward_hook(lambda m,i,o,n=n,idx=idx:saved.__setitem__(n,o[0,positions][:,idx.to(o.device)].detach().clone())))
    try:_run(model,ids)
    finally:
        for h in hooks:h.remove()
    return saved

def replace_hooks(targets,saved,positions,length):
    hooks=[]
    for n,(mod,idx) in enumerate(targets):
        def hook(m,i,o,n=n,idx=idx):
            if o.shape[1]!=length:return None
            o=o.clone();o[0,torch.tensor(positions,device=o.device)[:,None],idx.to(o.device)[None,:]]=saved[n].to(o.dtype);return o
        hooks.append(mod.register_forward_hook(hook))
    return hooks

def capture_keys(model,ids,module,layers,positions):
    """Post-RoPE keys (one per key/value group) at `positions` in `layers`, as the attention kernel receives them."""
    saved={};base=module.eager_attention_forward
    def wrapper(mod,query,key,value,attention_mask,*a,**kw):
        if mod.layer_idx in layers:saved[mod.layer_idx]=key[:,:,positions].detach().clone()
        return base(mod,query,key,value,attention_mask,*a,**kw)
    module.eager_attention_forward=wrapper
    try:_run(model,ids)
    finally:module.eager_attention_forward=base
    return saved

@contextlib.contextmanager
def head_keys(module,clean,heads,positions,length):
    """Only the query heads {layer:[head]} see clean keys at `positions`; their group-mates keep the blocked ones.

    Keys are expanded per query head (the repeat_kv layout: head h reads group h // n) and handed to the current kernel with
    num_key_value_groups = 1, so masks, soft-capping and the block stay the kernel's. Applies to every pass whose keys
    cover the prompt, candidate tails included (the KV cache itself is never modified)."""
    base=module.eager_attention_forward
    def wrapper(mod,query,key,value,attention_mask,*a,**kw):
        hs=heads.get(mod.layer_idx)
        if not hs or key.shape[-2]<length:return base(mod,query,key,value,attention_mask,*a,**kw)
        n=mod.num_key_value_groups;k=key.repeat_interleave(n,dim=1) if n>1 else key.clone();v=value.repeat_interleave(n,dim=1) if n>1 else value
        pos=torch.tensor(positions,device=k.device);ck=clean[mod.layer_idx].to(k.dtype)
        for h in hs:k[:,h,pos]=ck[:,h//n]
        mod.num_key_value_groups=1
        try:return base(mod,query,k,v,attention_mask,*a,**kw)
        finally:mod.num_key_value_groups=n
    module.eager_attention_forward=wrapper
    try:yield
    finally:module.eager_attention_forward=base

def held_out(cfg):
    held=mech_e4.make_items(dict(cfg,data_seed=cfg['head_seed']))
    return held[::max(1,len(held)//cfg['score_items'])][:cfg['score_items']]  # both splits, both modes

def run(cfg):
    cfg=dict(cfg);store=Store(cfg['out'],cfg);start=time.monotonic();deadline=start+cfg['budget_seconds'];status='COMPLETE';error=None
    items=mech_e4.make_items(cfg);atomic_json(store.root/'items.json',[{k:v for k,v in x.items() if k!='bundle'} for x in items])
    expected=0
    try:
        model,tok,identity=model_and_tokenizer(cfg,store);family=mech_v2.family_module(model)
        n_layers=len(model.model.layers);n_heads=model.config.num_attention_heads;L=cfg['path_layer']
        decompose=model.config.model_type in PRE_NORM
        if not 0<=L<n_layers:raise ValueError(f'path_layer {L} outside 0..{n_layers-1}')
        scoring=held_out(cfg)
        bind=mech_e4.binding_scores(model,tok,scoring,cfg['device']);read=mech_e4.reader_scores(model,tok,scoring,cfg['device'])
        depth=max(1,int(round(n_layers*cfg['head_depth'])));rng=random.Random(cfg['data_seed'])
        every=[(l,h) for l in range(n_layers) for h in range(n_heads)]
        upto=[(l,h) for l in range(L+1) for h in range(n_heads)];pool=[(l,h) for l in range(depth) for h in range(n_heads)]
        by=lambda s:(lambda lh:-float(s[lh[0],lh[1]]))
        ranked_upto=sorted(upto,key=by(bind));ranked=sorted(pool,key=by(bind));rread=sorted(every,key=by(read));readers=rread[:cfg['read_k']]
        sets={}  # variant -> (kind, heads)
        if decompose:
            for k in cfg['dec_ks']:sets[f'dec_top{k}']=('dec',ranked_upto[:k])
            for k in cfg['dec_rand_ks']:
                for r in range(cfg['random_draws']):sets[f'dec_rand{k}_{r}']=('dec',rng.sample(upto,min(k,len(upto))))
            sets[f'dec_top{cfg["bind_k"]}_mlp']=('dec_mlp',ranked_upto[:cfg['bind_k']])
        rk=cfg['read_k'];sets[f'kv_read{rk}']=('kv',readers);sets[f'k_read{rk}']=('k',readers);sets[f'v_read{rk}']=('v',readers)
        for r in range(cfg['random_draws']):sets[f'kv_rand{rk}_{r}']=('kv',rng.sample(every,rk))
        bk=cfg['bind_k'];sets[f'bind_top{bk}']=('bind',ranked[:bk])
        for r in range(cfg['random_draws']):sets[f'bind_rand{bk}_{r}']=('bind',rng.sample(pool,bk))
        # Round 6 sets are drawn after all earlier ones, so earlier random sets stay identical for a given seed.
        for k in cfg['head_ks']:sets[f'kh_read{k}']=('kh',rread[:k])
        per=n_heads//model.config.num_key_value_heads;mates=sorted({(l,g*per+j) for l,h in readers for g in [h//per] for j in range(per)}-set(readers))
        sets[f'kh_rest{rk}']=('kh',mates)
        for r in range(cfg['random_draws']):sets[f'kh_rand{rk}_{r}']=('kh',rng.sample(every,rk))
        parts=set(cfg.get('parts',['sender','receiver','route']))
        part={'dec':'sender','dec_mlp':'sender','kv':'receiver','k':'receiver','v':'receiver','bind':'route','kh':'heads'}
        variants=(['clean','blocked']+(['blocked_vals'] if 'route' in parts else [])+(['dec_all','dec_attn','dec_mlp'] if decompose and 'sender' in parts else [])
                  +[v for v,(kind,_) in sets.items() if part[kind] in parts and (kind not in ('dec','dec_mlp') or decompose)])
        mass_variants=(['clean','blocked','blocked_vals',f'bind_top{bk}']+[f'bind_rand{bk}_{r}' for r in range(cfg['random_draws'])]) if 'route' in parts else []
        atomic_json(store.root/'manifest.json',dict(version=__version__,source_sha256=source_hash(),config_sha256=store.fingerprint,model=identity,
                    n_layers=n_layers,n_heads=n_heads,n_kv_heads=model.config.num_key_value_heads,path_layer=L,decomposition=decompose,parts=sorted(parts),
                    head_scoring_items=[x['id'] for x in scoring],torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        atomic_json(store.root/'head_scores.json',dict(scores=bind.tolist(),reader_scores=read.tolist(),depth=depth,readers=[list(x) for x in readers],
                    reader_groups={str(l):g for l,g in kv_groups(model,readers).items()},head_sets={k:[list(x) for x in v[1]] for k,v in sets.items()},
                    kinds={k:v[0] for k,v in sets.items()}))
        expected=len(variants)*len(items)
        done={store.key(dict(id=r['id'],method=r['variant'],seed=0)) for r in store.load('path')}
        mass_done={store.key(dict(id=r['id'],method=r['variant'],seed=0)) for r in store.load('pathmass')}
        for item in items:
            lay=mech_e4.layout(tok,item);ids=mech_e4._ids(tok,item,lay,cfg['device']);S=lay['start'];later=lay['later'];cache={}
            def full_block():return mech_v2.blocked(later,lay['earlier'],S,cfg['device'],module=family)
            def comps():
                if 'comps' not in cache:
                    clean=components(model,ids,later,L)
                    with full_block() as st:
                        blocked=components(model,ids,later,L)
                        if st['applied']!=n_layers:raise RuntimeError('Block missed layers')
                    cache['comps']=(clean,blocked)
                return cache['comps']
            for variant in variants:
                key=dict(id=item['id'],method=variant,seed=0)
                if store.key(key) in done:continue
                if time.monotonic()>deadline:raise BudgetExceeded('Path budget exhausted')
                sync(cfg['device']);t0=time.monotonic();extra={};hooks=[];kind=sets[variant][0] if variant in sets else variant
                try:
                    if variant=='clean':scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length'])
                    elif kind=='bind' or variant=='blocked_vals':
                        heads=mech_e4._heads(sets[variant][1]) if kind=='bind' else None
                        with mech_v2.blocked(later,lay['earlier_value_tokens'],S,cfg['device'],module=family,heads=heads,n_heads=n_heads) as st:
                            scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length'])
                            if st['applied']!=(len(heads) if heads else n_layers):raise RuntimeError('Block missed layers')
                    else:
                        if variant.startswith('dec_') or kind in ('dec','dec_mlp'):
                            clean,blocked=comps()
                            if variant=='dec_all':add=delta(model,clean,blocked,'all',mlp=True)
                            elif variant=='dec_attn':add=delta(model,clean,blocked,'all')
                            elif variant=='dec_mlp':add=delta(model,clean,blocked,None,mlp=True)
                            else:add=delta(model,clean,blocked,mech_e4._heads(sets[variant][1]),mlp=kind=='dec_mlp')
                            if add is not None:hooks.append(add_hook(model.model.layers[L],later,add,S))
                            extra['path_layer']=L
                        elif kind in ('kv','k','v'):
                            groups=kv_groups(model,sets[variant][1]);targets=kv_targets(model,groups,'kv' if kind=='kv' else kind)
                            hooks+=replace_hooks(targets,capture(model,ids,targets,later),later,S)
                            extra.update(kv_groups=sum(len(g) for g in groups.values()),kv_layers=len(groups))
                        ctx=contextlib.nullcontext()
                        if kind=='kh':
                            if 'keys' not in cache:cache['keys']=capture_keys(model,ids,family,set(range(n_layers)),later)
                            heads=mech_e4._heads(sets[variant][1]);extra.update(query_heads=len(sets[variant][1]))
                        with full_block() as st:
                            if kind=='kh':ctx=head_keys(family,cache['keys'],heads,later,S)
                            with ctx:scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length'])
                            if st['applied']!=n_layers:raise RuntimeError('Block missed layers')
                finally:
                    for h in hooks:h.remove()
                row=dict(key,variant=variant,kind=kind,bundle_id=item['bundle_id'],split=item['split'],mode=item['mode'],condition=list(item['condition']),
                         latency_s=time.monotonic()-t0,**extra,**scored,**probe_v2.decide(item,scored))
                store.save('path',row);done.add(store.key(row))
            for variant in mass_variants:
                key=dict(id=item['id'],method=variant,seed=0)
                if store.key(key) in mass_done:continue
                if variant=='clean':A=mech_e4._attentions(model,ids)
                else:
                    if variant=='blocked':ctx=full_block()
                    else:
                        heads=mech_e4._heads(sets[variant][1]) if variant in sets else None
                        ctx=mech_v2.blocked(later,lay['earlier_value_tokens'],S,cfg['device'],module=family,heads=heads,n_heads=n_heads)
                    with ctx:A=mech_e4._attentions(model,ids)
                mass=mech_e4.reader_mass(A,lay);del A
                if mass is None:continue
                value=float(torch.stack([mass[l,h] for l,h in readers]).mean())
                store.save('pathmass',dict(key,variant=variant,bundle_id=item['bundle_id'],split=item['split'],mode=item['mode'],reader_mass=value))
                mass_done.add(store.key(key))
            print(f'PATH {len(done)}/{expected} elapsed={time.monotonic()-start:.0f}s',flush=True)
    except BudgetExceeded as e:status='BUDGET_EXHAUSTED';error=str(e)
    except Exception as e:status='RUNTIME_ERROR';error=repr(e);(store.root/'ERROR.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        rows=store.load('path');table={}
        for r in rows:table.setdefault((r['variant'],r['mode']),[]).append(r['correct'])
        write_csv(store.root/'path_by_variant.csv',[dict(variant=k[0],mode=k[1],n=len(v),accuracy=sum(v)/len(v)) for k,v in sorted(table.items())])
        atomic_json(store.root/'run_status.json',dict(status=status,error=error,completed=len(rows),expected=expected,elapsed_s=time.monotonic()-start))
    return dict(status=status,error=error,completed=len(rows),expected=expected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['path_smoke','mech_path'],default='mech_path')
    p.add_argument('--out',default='mech_path');p.add_argument('--config');a=p.parse_args()
    cfg=preset(a.preset)
    if a.config:cfg.update(json.loads(Path(a.config).read_text()))
    cfg['out']=str(Path(a.out).resolve());result=run(cfg);print(json.dumps(result,indent=2))
    if result['status']=='RUNTIME_ERROR':sys.exit(2)

if __name__=='__main__':main()
