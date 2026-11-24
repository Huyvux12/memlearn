"""Version-scoped Qwen3 intervention: native masks/cache/RoPE stay in HF.

All gates consume native post-RoPE Q/K. No task labels, token spans or gold.
The native eager function computes alpha; the trainable branch acts before W_O.
"""
import importlib
import math
import sys
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen3 import modeling_qwen3 as native

SUPPORTED=('STATIC','QUERY','KEY','SEPARABLE','SCALAR_PAIR','VECTOR_PAIR','POST_GATE',
           'RCA_ADAPT','DVA_ADAPT','AUX_ATTENTION','LORA_V','LORA_O','LORA_QK','LORA_QKVO','RESIDUAL')
_NATIVE_EAGER=native.eager_attention_forward
# Original eager kernel per HF modeling module (Qwen3 by default; Phi3 etc. registered on first attach).
_ORIGINALS={native:_NATIVE_EAGER}
SUPPORTED_ARCH=('qwen3','phi3')

def _family(obj):
    return sys.modules.get(type(obj).__module__) or importlib.import_module(type(obj).__module__)

def lora_targets(attn,method):
    fused=hasattr(attn,'qkv_proj')
    if fused and method in ('LORA_V','LORA_QK'):raise ValueError(method+' needs separate q/k/v projections; this architecture fuses them')
    return {'LORA_V':['v_proj'],'LORA_O':['o_proj'],'LORA_QK':['q_proj','k_proj'],
            'LORA_QKVO':['qkv_proj','o_proj'] if fused else ['q_proj','k_proj','v_proj','o_proj'],'RESIDUAL':['o_proj']}[method]

class ValueAdapter(nn.Module):
    def __init__(self,method,heads,dim,rank=8,chunk=16):
        super().__init__();self.method=method;self.chunk=chunk;self.rank=rank
        def matrix(*shape):return nn.Parameter(torch.randn(*shape)*(.15/math.sqrt(dim)))
        self.down=matrix(heads,dim,rank)
        self.up=nn.Parameter(torch.zeros(heads,rank,dim))
        if method not in ('STATIC','KEY','POST_GATE','RCA_ADAPT'):self.qmap=matrix(heads,dim,rank)
        if method not in ('STATIC','QUERY','POST_GATE','RCA_ADAPT'):self.kmap=matrix(heads,dim,rank)
        if method=='STATIC':self.bias=nn.Parameter(torch.zeros(heads,rank))
        if method=='RCA_ADAPT':
            del self.down;del self.up
            self.gamma=nn.Parameter(torch.zeros(heads,1,1))
        if method=='DVA_ADAPT':del self.down
    def forward(self,q,k,v,alpha,mask=None):
        dtype=v.dtype;q=q.float();k=k.float();v=v.float();alpha=alpha.float()
        m=self.method
        if m=='RCA_ADAPT':
            extra=F.softplus(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]))
            # Original RCA adds a positive unnormalized scalar reweighting. Signed
            # zero-init gamma here makes identity possible: explicitly an adaptation.
            return ((alpha*extra*self.gamma)@v).to(dtype)
        if m=='POST_GATE':
            gate=torch.einsum('bhtr,hrd->bhtd',torch.einsum('bhtd,hdr->bhtr',q,self.down),self.up)
            return ((alpha@v)*torch.tanh(gate)).to(dtype)
        z=torch.einsum('bhsd,hdr->bhsr',v,self.down) if hasattr(self,'down') else None
        qq=torch.einsum('bhtd,hdr->bhtr',q,self.qmap) if hasattr(self,'qmap') else None
        kk=torch.einsum('bhsd,hdr->bhsr',k,self.kmap) if hasattr(self,'kmap') else None
        if m=='STATIC':agg=(alpha@z)*(1+torch.tanh(self.bias))[None,:,None,:]
        elif m=='QUERY':agg=(alpha@z)*(1+torch.tanh(qq))
        elif m=='KEY':agg=alpha@(z*(1+torch.tanh(kk)))
        elif m=='SEPARABLE':agg=(alpha@(z*(1+torch.tanh(kk))))*(1+torch.tanh(qq))
        elif m=='DVA_ADAPT':
            # Low-rank residual dynamic value (DVA-inspired), separate from value scaling.
            agg=qq*(alpha@kk)
        elif m=='AUX_ATTENTION':
            logits=(qq@kk.transpose(-1,-2))/math.sqrt(self.rank)
            if mask is not None:logits=logits+mask[:,:,:,:k.shape[-2]].float()
            elif q.shape[-2]>1:raise ValueError('Missing causal mask for auxiliary prefill')
            agg=logits.softmax(-1)@z
        else:
            chunks=[]
            for start in range(0,q.shape[-2],self.chunk):
                stop=start+self.chunk;qt=qq[:,:,start:stop]
                if m=='VECTOR_PAIR':gate=1+torch.tanh(qt.unsqueeze(-2)+kk.unsqueeze(-3))
                elif m=='SCALAR_PAIR':gate=1+torch.tanh((qt.unsqueeze(-2)*kk.unsqueeze(-3)).sum(-1,keepdim=True)/math.sqrt(self.rank))
                else:raise ValueError(m)
                chunks.append(torch.einsum('bhts,bhtsr,bhsr->bhtr',alpha[:,:,start:stop],gate,z))
            agg=torch.cat(chunks,dim=-2)
        return torch.einsum('bhtr,hrd->bhtd',agg,self.up).to(dtype)

class LoRALinear(nn.Module):
    def __init__(self,base,rank,residual=False):
        super().__init__();self.base=base;self.residual=residual
        size=base.out_features if residual else base.in_features
        self.down=nn.Parameter(torch.randn(size,rank,device=base.weight.device)*(.15/math.sqrt(size)))
        self.up=nn.Parameter(torch.zeros(rank,base.out_features,device=base.weight.device))
    def forward(self,x):
        y=self.base(x);z=y if self.residual else x
        low=z.float()@self.down
        if self.residual:low=F.gelu(low)
        return y+(low@self.up).to(y.dtype)

class Handle:
    def __init__(self,model,method):self.model=model;self.method=method;self.modules={};self.restores=[]
    def named_parameters(self):
        for name,mod in self.modules.items():
            for k,p in mod.named_parameters():
                if p.requires_grad:yield name+'.'+k,p
    def parameters(self):return (p for _,p in self.named_parameters())
    def state_dict(self):return {k:p.detach().cpu().clone() for k,p in self.named_parameters()}
    def load_state_dict(self,state):
        current=dict(self.named_parameters())
        if current.keys()!=state.keys():raise ValueError('Adapter state keys mismatch')
        with torch.no_grad():
            for k,p in current.items():
                if p.shape!=state[k].shape:raise ValueError('Adapter shape mismatch')
                p.copy_(state[k].to(p.device))
    def remove(self):
        for parent,key,old in reversed(self.restores):
            if old is None:delattr(parent,key)
            else:setattr(parent,key,old)
        self.restores=[]
        self.model._memlearn_handle=None

def _dispatch(module,query,key,value,attention_mask,scaling,dropout=0.0,**kwargs):
    out,alpha=_ORIGINALS.get(_family(module),_NATIVE_EAGER)(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
    adapter=getattr(module,'memlearn_adapter',None)
    if adapter is None:return out,alpha
    k=native.repeat_kv(key,module.num_key_value_groups);v=native.repeat_kv(value,module.num_key_value_groups)
    delta=adapter(query,k,v,alpha,attention_mask)
    return out+delta.transpose(1,2).contiguous(),alpha

def attach(model,method,layers,rank=8,chunk=16):
    if transformers.__version__!='4.57.1':raise RuntimeError('Requires transformers==4.57.1; refusing untested attention API')
    if model.config.model_type not in SUPPORTED_ARCH:raise ValueError('Supported decoders: '+', '.join(SUPPORTED_ARCH))
    if method not in SUPPORTED:raise ValueError(method)
    if getattr(model,'_memlearn_handle',None):raise ValueError('Remove the existing adapter before attaching another')
    if model.config._attn_implementation!='eager':model.set_attn_implementation('eager')
    if not layers or len(set(layers))!=len(layers):raise ValueError('Select distinct layer indices')
    if any(i<0 or i>=len(model.model.layers) for i in layers):raise ValueError('Layer out of range')
    for p in model.parameters():p.requires_grad_(False);p.grad=None
    h=Handle(model,method)
    family=_family(model)
    if family.eager_attention_forward is not _dispatch:_ORIGINALS[family]=family.eager_attention_forward
    family.eager_attention_forward=_dispatch
    model._memlearn_handle=h
    try:
        for index in layers:
            layer=model.model.layers[index];attn=layer.self_attn
            if method.startswith('LORA_') or method=='RESIDUAL':
                for name in lora_targets(attn,method):
                    old=getattr(attn,name);mod=LoRALinear(old,rank,residual=method=='RESIDUAL')
                    setattr(attn,name,mod);h.restores.append((attn,name,old));h.modules[f'l{index}_{name}']=mod
            else:
                mod=ValueAdapter(method,model.config.num_attention_heads,attn.head_dim,rank,chunk).to(attn.o_proj.weight.device)
                attn.memlearn_adapter=mod;h.restores.append((attn,'memlearn_adapter',None));h.modules[f'l{index}']=mod
    except BaseException:
        h.remove();raise
    model._memlearn_handle=h
    return h
