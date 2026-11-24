"""Phase C screen on benchmark v2: train each adapter on book_first items, score with the log-prob probe.

Held-out axis: source order. Training never sees rule_first, where the frozen model fails to compose
codebook and rule (runs/probe_v2_kaggle/ANALYSIS.md). Pre-registered contrasts: docs/SCREEN_V2_PLAN.md.
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
import torch
from . import __version__
from . import data_v2,probe_v2
from .attention import attach,SUPPORTED
from .runner import Store,BudgetExceeded,atomic_json,layers_for,model_and_tokenizer,rank_for,source_hash,sync,write_csv

def preset(name):
    base=dict(name=name,benchmark='v2',model='Qwen/Qwen3-4B-Instruct-2507',revision='main',data_seed=17,seed=17,
              n_train=64,n_test=12,n_ood=12,train_steps=256,lr=.002,rank=8,method_ranks={},match_parameter_budget=True,layers='middle',chunk=16,
              methods=['FULL','QUERY','SEPARABLE','VECTOR_PAIR','LORA_V'],
              train_conditions=[[f,d,'book_first'] for f in ('neutral','default') for d in ('none','q')],
              eval_conditions=[['neutral',d,o] for d in ('none','q') for o in ('book_first','rule_first')],
              eval_modes=['READ_DEC','READ_RULE','APPLY_P','NONAPPLY'],
              dtype='float16',device='cuda',max_length=1024,gradient_checkpointing=True,budget_seconds=1500,
              out='screen_v2',execution_kind='PRETRAINED_QWEN')
    if name=='screen_smoke':
        base.update(model='RANDOM_TINY_QWEN',revision='TEST_ONLY',n_train=2,n_test=1,n_ood=1,train_steps=3,rank=2,layers=[0],
                    methods=['FULL','QUERY','VECTOR_PAIR','LORA_V'],dtype='float32',device='cpu',max_length=8192,
                    gradient_checkpointing=False,budget_seconds=600,execution_kind='RANDOM_QWEN_TEST_ONLY')
    elif name!='screen_v2':raise ValueError(name)
    return base

def train_example(tok,item,device,max_length):
    prompt=tok.encode(probe_v2.prompt_text(tok,item),add_special_tokens=False)
    # Separate target encoding keeps an exact prompt boundary; EOS teaches the answer to stop.
    target=tok.encode(data_v2.target(item['bundle'],item['mode']),add_special_tokens=False)+[tok.eos_token_id]
    if len(prompt)+len(target)>max_length:raise ValueError('Overlong training example; no silent truncation')
    return dict(input_ids=torch.tensor([prompt+target],device=device),labels=torch.tensor([[-100]*len(prompt)+target],device=device))

def train_items(ds,cfg):
    return data_v2.items({'train':ds['train']},('FULL',),[tuple(c) for c in cfg['train_conditions']],data_v2.MODES)

def train(model,tok,ds,method,cfg,store,deadline):
    """Returns an attached, trained handle. The caller must remove it."""
    seed=cfg['seed'];torch.manual_seed(seed);random.seed(seed)
    rank=rank_for(model,method,cfg);h=attach(model,method,layers_for(model,cfg),rank,cfg['chunk'])
    try:
        params=list(h.parameters());opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=0.)
        pool=train_items(ds,cfg);random.Random(seed).shuffle(pool)
        if cfg['gradient_checkpointing']:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train();initial=h.state_dict()
        for step in range(cfg['train_steps']):
            if time.monotonic()>deadline:raise BudgetExceeded('Training budget exhausted for '+method)
            example=train_example(tok,pool[step%len(pool)],cfg['device'],cfg['max_length'])
            t0=time.monotonic();opt.zero_grad(set_to_none=True)
            loss=model(**example,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);opt.step();sync(cfg['device'])
            store.save('training',dict(id=f'{method}_step_{step}',method=method,seed=seed,step=step+1,loss=float(loss.detach()),
                                       gradient_norm=float(norm),latency_s=time.monotonic()-t0,trainable_params=sum(p.numel() for p in params),
                                       peak_vram_bytes=torch.cuda.max_memory_allocated() if cfg['device']=='cuda' else None))
            if (step+1)%32==0:print(f'TRAIN {method} {step+1}/{cfg["train_steps"]} loss={float(loss):.4f}',flush=True)
        if not any(not torch.equal(initial[k],v) for k,v in h.state_dict().items()):raise RuntimeError('Adapter did not change')
        torch.save(dict(adapter=h.state_dict(),method=method,rank=rank,steps=cfg['train_steps']),store.root/f'adapter_{method}.pt')
        return h
    except BaseException:
        h.remove();raise
    finally:
        model.eval()
        if cfg['gradient_checkpointing']:model.gradient_checkpointing_disable()

def run(cfg):
    cfg=dict(cfg);store=Store(cfg['out'],cfg);start=time.monotonic();deadline=start+cfg['budget_seconds'];status='COMPLETE';error=None
    unknown=set(cfg['methods'])-set(SUPPORTED)-{'FULL'}
    if unknown:raise ValueError('Unknown methods '+str(unknown))
    ds=data_v2.splits(cfg['n_train'],cfg['n_test'],cfg['n_ood'],seed=cfg['data_seed'],style=cfg.get('style','codes'));atomic_json(store.root/'dataset_v2.json',ds)
    evaluation=data_v2.items({k:ds[k] for k in ('test_id','test_ood')},('FULL',),[tuple(c) for c in cfg['eval_conditions']],cfg['eval_modes'])
    availability={}
    try:
        model,tok,identity=model_and_tokenizer(cfg,store)
        atomic_json(store.root/'manifest.json',dict(version=__version__,source_sha256=source_hash(),config_sha256=store.fingerprint,model=identity,
                                                    torch=torch.__version__,python=sys.version,benchmark='v2',evaluation='teacher-forced candidate log-probabilities',
                                                    held_out='source_order rule_first',gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        for item in train_items(ds,cfg):train_example(tok,item,'cpu',cfg['max_length'])  # length check before spending GPU
        done={store.key(dict(id=r['id'],method=r['method'],seed=r['seed'])) for r in store.load('probe')}
        for method in cfg['methods']:
            pending=[x for x in evaluation if store.key(dict(id=x['id'],method=method,seed=cfg['seed'])) not in done]
            if not pending:availability[method]='DONE';continue
            if getattr(model,'_memlearn_handle',None) is not None:raise RuntimeError('Stale adapter before '+method)
            h=None;t0=time.monotonic()
            try:
                if method!='FULL':h=train(model,tok,ds,method,cfg,store,deadline)
                model.eval();params=sum(p.numel() for p in h.parameters()) if h else 0
                for item in pending:
                    if time.monotonic()>deadline:raise BudgetExceeded('Evaluation budget exhausted during '+method)
                    sync(cfg['device']);t1=time.monotonic()
                    scored=probe_v2.score_candidates(model,tok,item,cfg['device'],cfg['max_length']);sync(cfg['device'])
                    row=dict(id=item['id'],method=method,control=method,seed=cfg['seed'],bundle_id=item['bundle_id'],split=item['split'],family=item['family'],
                             mode=item['mode'],condition=list(item['condition']),trainable_params=params,latency_s=time.monotonic()-t1,
                             **scored,**probe_v2.decide(item,scored))
                    store.save('probe',row);done.add(store.key(row))
                availability[method]=dict(status='DONE',wall_s=time.monotonic()-t0,trainable_params=params)
                print(f'METHOD {method} done in {time.monotonic()-t0:.0f}s',flush=True)
            except BudgetExceeded as e:
                availability[method]=dict(status='BUDGET_EXHAUSTED',reason=str(e));raise
            except Exception as e:
                availability[method]=dict(status='ERROR',reason=repr(e));(store.root/f'ERROR_{method}.log').write_text(traceback.format_exc())
                print(traceback.format_exc(),flush=True)
                if cfg['device']=='cuda':torch.cuda.empty_cache()
            finally:
                if h is not None:h.remove()
                atomic_json(store.root/'method_availability.json',availability)
    except BudgetExceeded as e:status='BUDGET_EXHAUSTED';error=str(e)
    except Exception as e:status='RUNTIME_ERROR';error=repr(e);(store.root/'ERROR.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        rows=store.load('probe');table,jss=probe_v2.summarize(rows) if rows else ([],[])
        write_csv(store.root/'screen_by_cell.csv',table);write_csv(store.root/'screen_jss_v2.csv',jss)
        expected=len(evaluation)*len(cfg['methods'])
        if status=='COMPLETE' and len(rows)<expected:status='PARTIAL_METHOD_COVERAGE'
        atomic_json(store.root/'run_status.json',dict(status=status,error=error,completed=len(rows),expected=expected,elapsed_s=time.monotonic()-start))
    return dict(status=status,error=error,completed=len(rows),expected=expected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['screen_smoke','screen_v2'],default='screen_v2')
    p.add_argument('--out',default='screen_v2');p.add_argument('--config');a=p.parse_args()
    cfg=preset(a.preset)
    if a.config:cfg.update(json.loads(Path(a.config).read_text()))
    cfg['out']=str(Path(a.out).resolve());result=run(cfg);print(json.dumps(result,indent=2))
    if result['status']=='RUNTIME_ERROR':sys.exit(2)

if __name__=='__main__':main()
