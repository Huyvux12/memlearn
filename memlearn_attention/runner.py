"""Train frozen-Qwen attention modules and export independently scored panels."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
import torch
from . import __version__
from . import data, metrics
from .attention import attach,SUPPORTED

class BudgetExceeded(Exception):pass

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));os.replace(tmp,path)

def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hash():
    root=Path(__file__).parent
    return data.digest({p.name:sha_file(p) for p in sorted(root.glob('*.py'))})

class Store:
    def __init__(self,path,config):
        self.root=Path(path);self.root.mkdir(parents=True,exist_ok=True)
        self.fingerprint=data.digest(config);p=self.root/'config.json'
        if p.exists() and data.digest(json.loads(p.read_text()))!=self.fingerprint:
            raise ValueError('Configuration changed. Start a new output directory; refusing mixed receipts.')
        if not p.exists():atomic_json(p,config)
    def key(self,row):return data.digest([row.get('seed'),row.get('method'),row['id'],row.get('repeat',0)])
    def save(self,kind,row):
        p=self.root/kind/(self.key(row)+'.json');wrapped={'config_sha256':self.fingerprint,'sha256':data.digest(row),'row':row}
        if p.exists():
            if json.loads(p.read_text())!=wrapped:raise ValueError('Receipt collision or changed output')
            return False
        atomic_json(p,wrapped);return True
    def load(self,kind):
        rows=[]
        for p in sorted((self.root/kind).glob('*.json')):
            x=json.loads(p.read_text())
            if x['config_sha256']!=self.fingerprint or x['sha256']!=data.digest(x['row']):raise ValueError('Receipt integrity mismatch: '+p.name)
            rows.append(x['row'])
        return rows

def preset(name):
    base=dict(name=name,model='Qwen/Qwen3-4B-Instruct-2507',revision='main',data_seed=17,seeds=[17],
              n_train=24,n_dev=4,n_dev_ood=4,n_test=6,n_ood=6,max_train_steps=32,epochs=1,lr=.002,rank=8,
              method_ranks={},match_parameter_budget=True,layers='middle',chunk=16,max_length=1024,max_new_tokens=96,
              budget_seconds=1740,train_budget_fraction=.45,temperature=0.,reliability_repeats=1,
              gradient_checkpointing=True,dtype='float16',device='cuda',out='attention_results',bootstrap=1000,
              methods=['FULL','SCOPE_PROMPT','FACTS_MATCHED','QUERY','VECTOR_PAIR'],dev_requests=4,
              execution_kind='PRETRAINED_QWEN',shuffle_examples=False)
    if name=='smoke':
        base.update(model='RANDOM_TINY_QWEN',revision='TEST_ONLY',n_train=1,n_dev=1,n_dev_ood=1,n_test=1,n_ood=1,
                    methods=['FULL','QUERY','VECTOR_PAIR'],max_train_steps=2,rank=2,layers=[0],
                    max_length=4096,max_new_tokens=3,device='cpu',dtype='float32',budget_seconds=180,
                    gradient_checkpointing=False,bootstrap=50,dev_requests=0,execution_kind='RANDOM_QWEN_TEST_ONLY')
    elif name=='research':
        base.update(n_train=512,n_dev=128,n_dev_ood=64,n_test=256,n_ood=256,seeds=[17,29,43],max_train_steps=0,
                    epochs=2,budget_seconds=86400,train_budget_fraction=.6,
                    methods=['FULL','NATIVE_SDPA','SCOPE_PROMPT','FACTS_ONLY','FACTS_MATCHED','FACTS_MATCHED_ALT']+list(SUPPORTED),dev_requests=16)
    elif name=='pilot_v11':
        # v1.1 screen: every train example once, twice the test bundles, and plain LoRA-V to see
        # whether rule leakage appears under ordinary PEFT. Not a confirmatory design.
        base.update(max_train_steps=0,n_test=12,n_ood=12,shuffle_examples=True,
                    methods=['FULL','SCOPE_PROMPT','FACTS_MATCHED','QUERY','VECTOR_PAIR','LORA_V'])
    elif name!='pilot30':raise ValueError(name)
    return base

class TestByteTokenizer:
    """ONLY for random-Qwen software smoke. Never a pretrained tokenizer."""
    eos_token_id=1;pad_token_id=0
    def apply_chat_template(self,msgs,tokenize=False,add_generation_prompt=True,**kwargs):
        return '\n'.join(x['role']+': '+x['content'] for x in msgs)+'\nassistant: '
    def encode(self,text,add_special_tokens=False):return [b+2 for b in text.encode()]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        if not text.isascii():raise ValueError('Test tokenizer offsets require ASCII fixture')
        return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids,skip_special_tokens=True):return bytes([i-2 for i in ids if 2<=i<258]).decode(errors='replace')

def prompt_text(tokenizer,r,method='FULL'):
    control=method if method in ('SCOPE_PROMPT','FACTS_ONLY') else 'FULL'
    return tokenizer.apply_chat_template(data.messages(r,control),tokenize=False,add_generation_prompt=True,enable_thinking=False)

def prompt_tokens(tok,request,method='FULL'):
    text=prompt_text(tok,request,method)
    if method not in ('FACTS_MATCHED','FACTS_MATCHED_ALT'):return tok.encode(text,add_special_tokens=False)
    start=text.index('RULE_MAP for P:');stop=text.index('\n',start)
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded['input_ids']);offsets=encoded['offset_mapping']
    filler=tok.encode(' neutral archival filler ' if method=='FACTS_MATCHED' else ' unrelated placeholder passage ',add_special_tokens=False)
    if not filler:raise ValueError('Empty filler encoding')
    indices=[i for i,(a,b) in enumerate(offsets) if a<stop and b>start]
    if not indices:raise ValueError('Could not locate rule tokens for matched control')
    for j,i in enumerate(indices):ids[i]=filler[j%len(filler)]
    return ids

def training_example(tok,r,max_length,device):
    text=prompt_text(tok,r);ids=tok.encode(text,add_special_tokens=False)
    # Separate suffix encoding gives an exact prompt-prefix boundary; inference uses the same prefix.
    target=tok.encode(data.canonical(r['gold']),add_special_tokens=False)+[tok.eos_token_id]
    if len(ids)+len(target)>max_length:raise ValueError(f'Overlong training example {len(ids)+len(target)} > {max_length}; no silent truncation')
    return {'input_ids':torch.tensor([ids+target],device=device),
            'labels':torch.tensor([[-100]*len(ids)+target],device=device)}

def schedule(ds,methods,seeds,repeats=1):
    out=[]
    # Bundle-blocked, modes interleaved across methods. Early stop keeps paired blocks.
    n=max(len(ds['test_id']),len(ds['test_ood']))
    for index in range(n):
        for split in ('test_id','test_ood'):
            if index>=len(ds[split]):continue
            for seed in seeds:
                rotated=methods[(index+seed)%len(methods):]+methods[:(index+seed)%len(methods)]
                for repeat in range(repeats):
                    for req in data.requests_for(ds[split][index]):
                        for method in rotated:out.append(dict(request=req,method=method,seed=seed,repeat=repeat))
    return out

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()

def model_and_tokenizer(cfg,store):
    from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
    if cfg['execution_kind']=='RANDOM_QWEN_TEST_ONLY':
        torch.manual_seed(1);torch.set_num_threads(1)
        c=Qwen3Config(vocab_size=258,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=3,
                      num_key_value_heads=1,head_dim=8,max_position_embeddings=4096,attention_dropout=0.)
        model=Qwen3ForCausalLM(c);model.set_attn_implementation('eager')
        return model.eval(),TestByteTokenizer(),{'model':'RANDOM_TINY_QWEN','revision':'TEST_ONLY'}
    if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable. Enable GPU in Kaggle; refusing silent 4B CPU fallback.')
    from huggingface_hub import snapshot_download
    ident=store.root/'model_identity.json'
    old=json.loads(ident.read_text()) if ident.exists() else None
    revision=old['revision'] if old else cfg['revision']
    local=snapshot_download(cfg['model'],revision=revision,allow_patterns=['*.json','*.safetensors','*.model','*.jinja','merges.txt','vocab.json'],ignore_patterns=['consolidated*'],max_workers=2)
    resolved=Path(local).name
    identity={'model':cfg['model'],'revision':resolved}
    if old is not None and old!=identity:raise ValueError('Pretrained model revision changed on resume')
    atomic_json(ident,identity)
    dtype=getattr(torch,cfg['dtype']);tok=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(local,local_files_only=True,torch_dtype=dtype,attn_implementation='eager',trust_remote_code=False)
    # Probe and mechanism runs may use other decoders; adapter runs stay Qwen3-only (attach() also enforces it).
    if model.config.model_type!='qwen3' and not cfg.get('allow_any_arch'):raise ValueError('Expected Qwen3 architecture')
    return model.to(cfg['device']).eval(),tok,identity

def layers_for(model,cfg):
    return [len(model.model.layers)//2] if cfg['layers']=='middle' else list(cfg['layers'])

def rank_for(model,method,cfg):
    if method in cfg['method_ranks']:return cfg['method_ranks'][method]
    if not cfg['match_parameter_budget']:return cfg['rank']
    from .attention import lora_targets
    c=model.config;attn=model.model.layers[0].self_attn;h=c.num_attention_heads;d=attn.head_dim;hidden=c.hidden_size
    factor={'STATIC':h*(2*d+1),'QUERY':3*h*d,'KEY':3*h*d,'SEPARABLE':4*h*d,
            'SCALAR_PAIR':4*h*d,'VECTOR_PAIR':4*h*d,'POST_GATE':2*h*d,
            'DVA_ADAPT':3*h*d,'AUX_ATTENTION':4*h*d,'RESIDUAL':2*hidden}
    if method.startswith('LORA_'):
        # LoRA cost per rank = in+out of each targeted Linear; covers fused qkv_proj (Phi3) and separate q/k/v (Qwen3).
        factor[method]=sum(getattr(attn,n).in_features+getattr(attn,n).out_features for n in lora_targets(attn,method))
    if method=='RCA_ADAPT':return cfg['rank'] # scalar reference cannot be matched by changing feature rank
    target=4*h*d*cfg['rank']
    return min(128,max(1,round(target/factor[method])))

def validate_config(cfg):
    allowed=set(preset('pilot30'))
    if set(cfg)-allowed:raise ValueError('Unknown configuration keys: '+str(set(cfg)-allowed))
    if len(set(cfg['methods']))!=len(cfg['methods']) or not cfg['methods']:raise ValueError('Methods must be nonempty and distinct')
    known=set(SUPPORTED)|{'FULL','NATIVE_SDPA','SCOPE_PROMPT','FACTS_ONLY','FACTS_MATCHED','FACTS_MATCHED_ALT'}
    if set(cfg['methods'])-known:raise ValueError('Unknown methods')
    if cfg['rank']<1 or cfg['chunk']<1 or cfg['max_new_tokens']<1 or cfg['max_length']<1:raise ValueError('Positive rank/chunk/token limits required')
    if cfg['budget_seconds']<1 or not 0<cfg['train_budget_fraction']<1:raise ValueError('Invalid time budget')
    if cfg['temperature']<0 or cfg['reliability_repeats']<1:raise ValueError('Invalid decoding settings')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('Distinct training seeds required')
    for key in ('n_train','n_dev','n_dev_ood','n_test','n_ood','max_train_steps'):
        if cfg[key]<0:raise ValueError('Negative dataset/step size')
    if cfg['n_train']==0 and set(cfg['methods'])&set(SUPPORTED):raise ValueError('Trainable methods require training examples')
    if cfg['epochs']<1 or cfg['lr']<=0:raise ValueError('Invalid training hyperparameters')
    if any(x<1 for x in cfg['method_ranks'].values()):raise ValueError('Invalid per-method rank')

def checkpoint_path(store,method,seed):return store.root/'checkpoints'/f'{method}_{seed}.json'

def save_checkpoint(path,payload):
    # Blob first, then one atomic pointer commit. Kill at either point leaves the
    # previous checkpoint usable; an orphan blob is harmless and not a result.
    path.parent.mkdir(parents=True,exist_ok=True)
    directory=path.parent/(path.stem+'_objects');directory.mkdir(exist_ok=True)
    tmp=directory/'pending.tmp';torch.save(payload,tmp);sha=sha_file(tmp)
    blob=directory/(sha+'.pt');os.replace(tmp,blob)
    previous=json.loads(path.read_text()).get('file') if path.exists() else None
    atomic_json(path,{'file':str(blob.relative_to(path.parent)),'sha256':sha})
    keep={str(blob.relative_to(path.parent)),previous}
    for old in directory.glob('*.pt'):
        if str(old.relative_to(path.parent)) not in keep:old.unlink()

def read_checkpoint(path,store):
    pointer=json.loads(path.read_text());blob=(path.parent/pointer['file']).resolve()
    if path.parent.resolve() not in blob.parents:raise ValueError('Checkpoint path escapes run directory')
    if sha_file(blob)!=pointer['sha256']:raise ValueError('Checkpoint hash mismatch')
    payload=torch.load(blob,map_location='cpu',weights_only=True)
    if payload['config_sha256']!=store.fingerprint:raise ValueError('Checkpoint configuration mismatch')
    return payload

def train_method(model,tok,ds,method,seed,cfg,store,deadline):
    if getattr(model,'_memlearn_handle',None) is not None:raise RuntimeError('Stale adapter before training')
    try:return _train_method_impl(model,tok,ds,method,seed,cfg,store,deadline)
    finally:
        model.eval()
        if cfg['gradient_checkpointing']:model.gradient_checkpointing_disable()
        active=getattr(model,'_memlearn_handle',None)
        if active is not None:active.remove()

def _train_method_impl(model,tok,ds,method,seed,cfg,store,deadline):
    rank=rank_for(model,method,cfg);torch.manual_seed(seed);random.seed(seed)
    h=attach(model,method,layers_for(model,cfg),rank,cfg['chunk']);params=list(h.parameters())
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=0.)
    path=checkpoint_path(store,method,seed);start_step=0
    train_bundles=list(ds['train']);random.Random(seed).shuffle(train_bundles)
    examples=[r for _ in range(cfg['epochs']) for b in train_bundles for r in data.requests_for(b)]
    # Bundle-blocked READ->APPLY->NONAPPLY->MIXED order lets a short prefix end on one mode; seeded shuffle keeps resume exact.
    if cfg['shuffle_examples']:random.Random(seed).shuffle(examples)
    total=min(len(examples),cfg['max_train_steps'] or len(examples))
    if path.exists():
        saved=read_checkpoint(path,store);h.load_state_dict(saved['adapter']);opt.load_state_dict(saved['optimizer']);start_step=saved['step']
        if saved.get('last_log'):store.save('training',saved['last_log'])
    if cfg['gradient_checkpointing']:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();initial=h.state_dict();changed=False
    try:
        for step in range(start_step,total):
            if time.monotonic()>deadline:raise BudgetExceeded('Training allocation exhausted')
            example=training_example(tok,examples[step],cfg['max_length'],cfg['device'])
            started=time.monotonic();opt.zero_grad(set_to_none=True)
            if cfg['device']=='cuda':torch.cuda.reset_peak_memory_stats()
            loss=model(**example,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite train loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('Non-finite adapter gradient')
            opt.step();sync(cfg['device'])
            changed=any(not torch.equal(initial[k],v) for k,v in h.state_dict().items())
            log=dict(id=f'step_{step}',method=method,seed=seed,step=step+1,total_steps=total,loss=float(loss.detach()),
                     gradient_norm=float(norm),adapter_changed=changed,trainable_params=sum(p.numel() for p in params),
                     latency_s=time.monotonic()-started,input_tokens=example['input_ids'].numel(),
                     peak_vram_bytes=torch.cuda.max_memory_allocated() if cfg['device']=='cuda' else None)
            save_checkpoint(path,dict(adapter=h.state_dict(),optimizer=opt.state_dict(),step=step+1,total_steps=total,
                                     config_sha256=store.fingerprint,last_log=log,method=method,seed=seed,rank=rank))
            store.save('training',log)
            print(f'TRAIN {method} seed={seed} {step+1}/{total} loss={log["loss"]:.4f}',flush=True)
        # Development is diagnostic only: last fixed checkpoint, never select on test.
        model.eval()
        for split in ('dev_id','dev_ood'):
            for req in [r for b in ds[split] for r in data.requests_for(b)][:cfg['dev_requests']]:
                key=dict(id=req['id'],method=method,seed=seed)
                if store.key(key) in {store.key(x) for x in store.load('development')}:continue
                if time.monotonic()>deadline:break
                with torch.no_grad():loss=model(**training_example(tok,req,cfg['max_length'],cfg['device']),use_cache=False).loss
                store.save('development',dict(key,split=split,mode=req['mode'],nll=float(loss)))
    finally:
        model.eval()
        if cfg['gradient_checkpointing']:model.gradient_checkpointing_disable()
        h.remove()
    return dict(method=method,seed=seed,steps=total,changed=changed)

def generate(model,tok,request,method,cfg,deadline,repeat=0,seed=0):
    text=prompt_text(tok,request,method);ids=prompt_tokens(tok,request,method)
    if len(ids)>cfg['max_length']:raise ValueError(f'Overlong prompt {len(ids)} > {cfg["max_length"]}')
    if time.monotonic()>deadline:raise BudgetExceeded('Evaluation budget exhausted')
    x=torch.tensor([ids],device=cfg['device']);generated=[]
    torch.manual_seed(int(data.digest([seed,repeat,request['id']])[:12],16)%2**31)
    sync(cfg['device']);started=time.monotonic()
    allocated_before=torch.cuda.memory_allocated() if cfg['device']=='cuda' else None
    if cfg['device']=='cuda':torch.cuda.reset_peak_memory_stats()
    eos=tok.eos_token_id;cache=None;prefill=None
    with torch.no_grad():
        for step in range(cfg['max_new_tokens']):
            if time.monotonic()>deadline:raise BudgetExceeded('Interrupted mid-generation; no scored receipt written')
            result=model(input_ids=x,past_key_values=cache,use_cache=True,logits_to_keep=1)
            logits=result.logits[:,-1,:].float();cache=result.past_key_values
            if step==0:sync(cfg['device']);prefill=time.monotonic()-started
            if cfg['temperature']>0:token=torch.multinomial((logits/cfg['temperature']).softmax(-1),1)
            else:token=logits.argmax(-1,keepdim=True)
            value=int(token.item());generated.append(value)
            if value==eos:break
            x=token
    sync(cfg['device']);elapsed=time.monotonic()-started
    output=tok.decode(generated,skip_special_tokens=True)
    return dict(output=output,input_tokens=len(ids),output_tokens=len(generated),latency_s=elapsed,
                prefill_s=prefill,decode_s=max(0.,elapsed-prefill),decode_tokens_per_s=(len(generated)-1)/max(1e-9,elapsed-prefill),
                peak_incremental_vram_bytes=torch.cuda.max_memory_allocated()-allocated_before if cfg['device']=='cuda' else None,
                peak_vram_bytes=torch.cuda.max_memory_allocated() if cfg['device']=='cuda' else None,
                termination='EOS' if generated[-1]==eos else 'MAX_NEW_TOKENS',prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),input_ids_sha256=data.digest(ids))

def algebra_checks():
    from .attention import ValueAdapter
    torch.manual_seed(719)
    q=torch.randn(1,2,3,4);k=torch.randn(1,2,5,4);v=torch.randn_like(k);alpha=torch.randn(1,2,3,5).softmax(-1)
    result={}
    for method in ('QUERY','SEPARABLE'):
        g=ValueAdapter(method,2,4,3,chunk=1)
        with torch.no_grad():g.up.normal_()
        z=torch.einsum('bhsd,hdr->bhsr',v,g.down);phi=1+torch.tanh(torch.einsum('bhtd,hdr->bhtr',q,g.qmap))
        if method=='SEPARABLE':z=z*(1+torch.tanh(torch.einsum('bhsd,hdr->bhsr',k,g.kmap)))
        expected=torch.einsum('bhtr,hrd->bhtd',phi*(alpha@z),g.up)
        err=float((g(q,k,v,alpha)-expected).abs().max().detach());result[method+'_equivalence_max_error']=err
        if err>1e-4:raise ValueError('Algebra check failed')
    gain=1+torch.sigmoid(q@k.transpose(-1,-2));weighted=alpha*gain;mass=weighted.sum(-1,keepdim=True)
    scalar_error=float(((weighted@v)-((weighted/mass)@v)*mass).abs().max())
    result['scalar_attention_plus_gain_max_error']=scalar_error
    if scalar_error>1e-5:raise ValueError('Scalar equivalence failed')
    g=1+torch.tanh(torch.tensor([[.1],[1.3]])+torch.tensor([[-.6,.9]]))
    determinant=float(torch.linalg.det(g));result['one_feature_gate_rank1_minor']=determinant
    if abs(determinant)<1e-4:raise ValueError('Nonseparable gate witness degenerate')
    result['interpretation']='Query and separable operators have exact post/factorized equivalents. Full pairwise gate comparison requires trained controls; no native-circuit claim.'
    return result

def write_csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def reliability(rows,k=3):
    groups={}
    for row in rows:
        key=tuple(row[x] for x in ('method','seed','split','bundle_id','mode'))
        groups.setdefault(key,{})[row.get('repeat',0)]=row['correct']
    eligible=[list(x.values())[:k] for x in groups.values() if len(x)>=k]
    by_method={}
    for key,values in groups.items():
        if len(values)<k:continue
        by_method.setdefault(key[:3],[]).append([values[i] for i in sorted(values)[:k]])
    panels=[dict(method=key[0],seed=key[1],split=key[2],eligible_instances=len(items),
                 pass_at_k_empirical=metrics.mean([int(any(x)) for x in items]),
                 pass_power_k_empirical=metrics.mean([int(all(x)) for x in items])) for key,items in by_method.items()]
    return dict(by_method=panels,k=k,eligible_instances=len(eligible),pass_at_k_empirical=metrics.mean([int(any(x)) for x in eligible]),
                pass_power_k_empirical=metrics.mean([int(all(x)) for x in eligible]),
                interpretation='Repeated same-checkpoint stochastic trials required; not product of aggregate accuracy. Greedy repeats are not independent.')

def export(store,cfg,algebra=None):
    rows=store.load('predictions');training=store.load('training');dev=store.load('development')
    panels=metrics.summarize(rows)
    from collections import defaultdict
    groups=defaultdict(list)
    for row in rows:groups[(row['method'],row['seed'],row['split'],row.get('family'),row['mode'],row.get('source_order'))].append(row)
    conditions=[]
    for (method,seed,split,family,mode,position),items in groups.items():
        conditions.append(dict(method=method,seed=seed,split=split,family=family,mode=mode,source_order=position,n=len(items),
                               accuracy=metrics.mean([x['correct'] for x in items]),
                               rule_leakage=metrics.mean([x.get('rule_leakage',0) for x in items]) if mode=='NONAPPLY' else None))
    write_csv(store.root/'metrics_by_condition.csv',conditions)
    train_groups=defaultdict(list)
    for row in training:train_groups[(row['method'],row['seed'])].append(row)
    costs=[]
    for (method,seed),items in train_groups.items():
        items=sorted(items,key=lambda x:x['step'])
        costs.append(dict(method=method,seed=seed,optimizer_steps=len(items),trainable_params=items[-1]['trainable_params'],
                          train_compute_s=sum(x['latency_s'] for x in items),train_tokens=sum(x['input_tokens'] for x in items),
                          first_loss=items[0]['loss'],last_loss=items[-1]['loss'],
                          peak_vram_bytes=max([x['peak_vram_bytes'] for x in items if x['peak_vram_bytes'] is not None],default=None)))
    write_csv(store.root/'training_cost.csv',costs)
    write_csv(store.root/'metrics_by_method.csv',panels);write_csv(store.root/'predictions.csv',rows)
    write_csv(store.root/'training.csv',training);write_csv(store.root/'development.csv',dev)
    comparisons=[]
    present=sorted(set(r['method'] for r in rows))
    for a in present:
        for b in ('FULL','QUERY','SEPARABLE','SCALAR_PAIR','POST_GATE','LORA_V','SCOPE_PROMPT'):
            if a==b or b not in present:continue
            for mode in (None,'READ','APPLY','NONAPPLY','MIXED'):
                for split in ('test_id','test_ood'):
                    comparisons.append(metrics.contrast(rows,a,b,bootstrap=cfg['bootstrap'],mode=mode,split=split))
            for split in ('test_id','test_ood'):
                comparisons.append(metrics.jss_contrast(rows,a,b,bootstrap=cfg['bootstrap'],split=split))
    write_csv(store.root/'paired_contrasts.csv',comparisons)
    rq=metrics.research_questions(rows,algebra or {},cfg['n_test']+cfg['n_ood'])
    scientific=cfg['execution_kind']=='PRETRAINED_QWEN'
    if not scientific:
        for value in rq.values():value['supported_claim']='TEST_ONLY: random Qwen validates software, not pretrained scientific efficacy.'
    atomic_json(store.root/'validity_by_rq.json',rq)
    atomic_json(store.root/'reliability.json',reliability(rows,cfg['reliability_repeats']))
    expected=(cfg['n_test']+cfg['n_ood'])*4*len(cfg['methods'])*len(cfg['seeds'])*cfg['reliability_repeats']
    audit={'receipts_verified':len(rows)+len(training)+len(dev),'completed_predictions':len(rows),'expected_predictions':expected,
           'schedule_complete':len(rows)==expected,'scientific_pretrained_results':scientific and bool(rows),
           'execution_kind':cfg['execution_kind'],'software_integrity':'CHECKSUMS_VERIFIED',
           'warning':'Checksums are integrity checks, not proof of scientific validity. Independent scoring audit is separate.'}
    coverage=[]
    expected_each=(cfg['n_test']+cfg['n_ood'])*4*cfg['reliability_repeats']
    for method in cfg['methods']:
        for seed in cfg['seeds']:
            count=sum(r['method']==method and r['seed']==seed for r in rows)
            coverage.append({'method':method,'seed':seed,'completed':count,'expected':expected_each,
                             'status':'COMPLETE' if count==expected_each else 'PARTIAL' if count else 'NOT_RUN'})
    write_csv(store.root/'coverage.csv',coverage)
    # Between-seed variation is separate from bundle bootstrap uncertainty.
    spread=[]
    for method in cfg['methods']:
        for split in ('test_id','test_ood'):
            vals=[p['jss'] for p in panels if p['method']==method and p['split']==split and p['jss'] is not None]
            spread.append({'method':method,'split':split,'seeds_observed':len(vals),'jss_mean':metrics.mean(vals),
                           'jss_seed_std':__import__('statistics').stdev(vals) if len(vals)>1 else None})
    write_csv(store.root/'seed_variation.csv',spread)
    atomic_json(store.root/'audit.json',audit)
    lines=['# MEMLEARN Attention Lab — measured results',f'Execution: {cfg["execution_kind"]}',f'Completed predictions: {len(rows)}/{expected}. Missing methods/requests are NOT_RUN, never scored as zero.',
           'No binary paper-level verdict is inferred from this pilot. Negative results are valid.',
           '| Method | Seed | Split | N | READ | APPLY | NONAPPLY | JSS | MIXED | Leakage |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'N/A' if x is None else f'{x:.3f}'
    for p in panels:
        lines.append(f'| {p["method"]} | {p["seed"]} | {p["split"]} | {p["n"]} | '+ ' | '.join(fmt(p.get(k)) for k in ('read_accuracy','apply_accuracy','nonapply_accuracy','jss','mixed_accuracy','rule_leakage_rate'))+' |')
    for key,value in rq.items():lines+=['',f'**{key}** — {value["coverage"]}',value['supported_claim']]
    lines+=['','Paper operators are local adaptations: see docs/BASELINES.md. No claims of official leaderboard reproduction.',
            'Causal localization, dialogue hysteresis, natural-task external transfer and a second pretrained model are not established by this release.',
            'Inspect paired_contrasts.csv, training.csv, raw receipts and validity_by_rq.json; do not interpret an absent CI/pair as no effect.']
    (store.root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return audit

def run(cfg):
    cfg=dict(cfg);validate_config(cfg);store=Store(cfg['out'],cfg);start=time.monotonic();deadline=start+cfg['budget_seconds'];status='COMPLETE';error=None
    code=source_hash();manifest_path=store.root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['source_sha256']!=code:raise ValueError('Source code changed on resume. Use a new output directory.')
    ds=data.dataset(cfg['n_train'],cfg['n_dev'],cfg['n_test'],cfg['n_ood'],seed=cfg['data_seed'],n_dev_ood=cfg['n_dev_ood']);atomic_json(store.root/'dataset.json',ds)
    algebra=algebra_checks();atomic_json(store.root/'algebra_checks.json',algebra)
    try:
        model,tok,identity=model_and_tokenizer(cfg,store)
        atomic_json(manifest_path,dict(version=__version__,source_sha256=code,config_sha256=store.fingerprint,model=identity,
                                      python=sys.version,torch=torch.__version__,platform=platform.platform(),
                                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                      operator='Qwen3 native eager attention + fp32 residual adapters',
                                      split_sha256=data.digest(ds),evaluation_mode='independent query completions; no dialogue carryover'))
        # Verify all prompt/target lengths BEFORE expensive training. No admission based on success.
        for bs in ds.values():
            for b in bs:
                for req in data.requests_for(b):training_example(tok,req,cfg['max_length'],'cpu')
        train_deadline=start+cfg['budget_seconds']*cfg['train_budget_fraction']
        availability={}
        for seed in cfg['seeds']:
            trainables=[m for m in cfg['methods'] if m in SUPPORTED]
            # Counterbalance which method trains first across seeds.
            if cfg['seeds'].index(seed)%2:trainables.reverse()
            for method in trainables:
                try:
                    phase_started=time.monotonic();info=train_method(model,tok,ds,method,seed,cfg,store,train_deadline)
                    availability[f'{method}/{seed}']={'status':'TRAINED','wall_s':time.monotonic()-phase_started,**info}
                except BudgetExceeded as e:
                    availability[f'{method}/{seed}']={'status':'TRAINING_INCOMPLETE','reason':str(e)}
                    print(str(e),flush=True);break
                except Exception as e:
                    availability[f'{method}/{seed}']={'status':'TRAIN_ERROR','reason':repr(e)}
                    (store.root/f'TRAIN_ERROR_{method}_{seed}.log').write_text(traceback.format_exc())
                    if cfg['device']=='cuda':torch.cuda.empty_cache()
                finally:atomic_json(store.root/'method_availability.json',availability)
        existing={store.key(r) for r in store.load('predictions')};states={}
        for item in schedule(ds,cfg['methods'],cfg['seeds'],cfg['reliability_repeats']):
            req=item['request'];method=item['method'];seed=item['seed'];repeat=item['repeat']
            key=dict(id=req['id'],method=method,seed=seed,repeat=repeat)
            if store.key(key) in existing:continue
            if time.monotonic()>deadline:raise BudgetExceeded('Evaluation allocation exhausted')
            if getattr(model,'_memlearn_handle',None) is not None:raise RuntimeError('Unexpected adapter before evaluation; refusing mislabeled baseline')
            h=None;training_steps=0;params=0
            if method in SUPPORTED:
                if availability.get(f'{method}/{seed}',{}).get('status')=='TRAIN_ERROR':continue
                cachekey=(method,seed)
                if cachekey not in states:
                    path=checkpoint_path(store,method,seed)
                    if not path.exists():continue
                    states[cachekey]=read_checkpoint(path,store)
                saved=states[cachekey]
                # Incomplete training is not silently promoted to the intended method.
                if saved['step']!=saved['total_steps']:continue
                h=attach(model,method,layers_for(model,cfg),saved['rank'],cfg['chunk']);h.load_state_dict(saved['adapter'])
                training_steps=saved['step'];params=sum(p.numel() for p in h.parameters())
            elif method not in ('FULL','NATIVE_SDPA','SCOPE_PROMPT','FACTS_ONLY','FACTS_MATCHED','FACTS_MATCHED_ALT'):raise ValueError('Unknown method '+method)
            try:
                model.set_attn_implementation('sdpa' if method=='NATIVE_SDPA' else 'eager')
                model.eval();out=generate(model,tok,req,method,cfg,deadline,repeat,seed)
                row=dict(key,bundle_id=req['bundle_id'],split=req['split'],family=req['family'],source_order=req['bundle']['source_order'],mode=req['mode'],
                         trainable_params=params,training_steps=training_steps,**out,**metrics.score(req,out['output']))
                store.save('requests',dict(id=req['id'],method=method,seed=seed,repeat=repeat,request=req,
                                           prompt=prompt_text(tok,req,method),input_ids=prompt_tokens(tok,req,method),decoded_input=tok.decode(prompt_tokens(tok,req,method),skip_special_tokens=False)))
                store.save('predictions',row);existing.add(store.key(key))
                print(f'EVAL {method} {req["split"]} {req["mode"]} success={row["correct"]} ({len(existing)})',flush=True)
                if len(existing)%12==0:atomic_json(store.root/'progress.json',{'completed_predictions':len(existing),'elapsed_s':time.monotonic()-start})
            finally:
                if h is not None:h.remove()
                model.set_attn_implementation('eager')
        expected_count=(cfg['n_test']+cfg['n_ood'])*4*len(cfg['methods'])*len(cfg['seeds'])*cfg['reliability_repeats']
        if len(existing)<expected_count:
            status='PARTIAL_METHOD_COVERAGE';error='Some trained methods are unavailable/incomplete; inspect method_availability.json and coverage.csv'
    except BudgetExceeded as e:status='BUDGET_EXHAUSTED';error=str(e)
    except KeyboardInterrupt:status='INTERRUPTED';error='User interrupted; completed receipts preserved'
    except Exception as e:
        status='RUNTIME_ERROR';error=repr(e);(store.root/'ERROR.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        atomic_json(store.root/'run_status.json',dict(status=status,error=error,elapsed_s=time.monotonic()-start))
        result=export(store,cfg,algebra)
    result.update(status=status,error=error,execution_kind=cfg['execution_kind'])
    return result

def audit_run(path):
    root=Path(path);cfg=json.loads((root/'config.json').read_text());s=Store(root,cfg)
    rows=s.load('predictions');requests={s.key(x):x for x in s.load('requests')};n=0
    for row in rows:
        stored=requests[s.key(row)];req=stored['request']
        if data.oracle(req['bundle'],req['mode'])!=req['gold']:raise ValueError('Stored oracle changed')
        rescored=metrics.score(req,row['output'])
        if any(row[k]!=v for k,v in rescored.items()):raise ValueError('Saved metric differs from replay')
        if hashlib.sha256(stored['prompt'].encode()).hexdigest()!=row['prompt_sha256']:raise ValueError('Prompt hash differs')
        if data.digest(stored['input_ids'])!=row['input_ids_sha256']:raise ValueError('Input token hash differs')
        if len(stored['input_ids'])!=row['input_tokens']:raise ValueError('Input token length differs')
        n+=1
    value={'replayed_predictions':n,'status':'VERIFIED','warning':'Replay validates stored output scoring, not independent authentication of neural outputs.'}
    atomic_json(root/'replay_audit.json',value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','pilot30','pilot_v11','research'],default='pilot30');p.add_argument('--out',default='attention_results');p.add_argument('--config');p.add_argument('--audit',action='store_true');p.add_argument('--export',action='store_true')
    a=p.parse_args()
    if a.audit:print(json.dumps(audit_run(a.out),indent=2));return
    if a.export:
        cfg=json.loads((Path(a.out)/'config.json').read_text());print(export(Store(a.out,cfg),cfg,json.loads((Path(a.out)/'algebra_checks.json').read_text())));return
    cfg=preset(a.preset)
    if a.config:cfg.update(json.loads(Path(a.config).read_text()))
    cfg['out']=str(Path(a.out).resolve());result=run(cfg);print(json.dumps(result,indent=2))
    if result['status']=='RUNTIME_ERROR':sys.exit(2)

if __name__=='__main__':main()
