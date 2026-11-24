import importlib.util
import pytest
import torch

torch.set_num_threads(1)

def api():
    assert importlib.util.find_spec('memlearn_attention.attention'), 'attention module missing'
    from memlearn_attention import attention
    return attention

def tiny():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(42)
    c=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                  num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
                  attention_dropout=0.0,tie_word_embeddings=False)
    model=Qwen3ForCausalLM(c).eval();model.set_attn_implementation('eager');return model

@pytest.mark.parametrize('method',['STATIC','QUERY','KEY','SEPARABLE','SCALAR_PAIR','VECTOR_PAIR','POST_GATE','RCA_ADAPT','DVA_ADAPT','AUX_ATTENTION','LORA_V','LORA_O','LORA_QK','LORA_QKVO','RESIDUAL'])
def test_identity_frozen_gradient_and_restore(method):
    a=api(); model=tiny();x=torch.tensor([[3,4,5,6,7]])
    with torch.no_grad():native=model(x).logits
    h=a.attach(model,method,[0],rank=3,chunk=2)
    actual=model(x).logits
    assert torch.allclose(actual,native,atol=1e-7,rtol=1e-6)
    loss=model(x,labels=x,use_cache=False).loss;loss.backward()
    params=list(h.parameters())
    assert params and any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    ids={id(p) for p in params}
    assert all(p.grad is None for p in model.parameters() if id(p) not in ids)
    with torch.no_grad():
        for p in params:
            if p.grad is not None:p.add_(-0.1*p.grad)
    h.remove()
    with torch.no_grad():restored=model(x).logits
    assert torch.equal(restored,native)

@pytest.mark.parametrize('method',['VECTOR_PAIR','QUERY','SEPARABLE','SCALAR_PAIR','RCA_ADAPT','DVA_ADAPT','AUX_ATTENTION','LORA_V'])
def test_cache_and_no_future_leak_with_nonzero_intervention(method):
    a=api();model=tiny();h=a.attach(model,method,[0],rank=3,chunk=2)
    with torch.no_grad():
        for p in h.parameters():p.add_(torch.randn_like(p)*.04)
        x=torch.tensor([[4,8,9,2,5]])
        full=model(x,use_cache=False).logits
        prefix=model(x[:,:3],use_cache=True)
        suffix=model(x[:,3:],past_key_values=prefix.past_key_values,use_cache=True).logits
        assert torch.allclose(full[:,3:],suffix,atol=2e-6,rtol=2e-5)
        altered=x.clone();altered[:,-1]=11
        other=model(altered,use_cache=False).logits
        assert torch.allclose(full[:,:-1],other[:,:-1],atol=2e-6,rtol=2e-5)
    h.remove()

def test_query_and_separable_exact_algebra_and_chunk_invariance():
    a=api();torch.manual_seed(3)
    q=torch.randn(2,4,5,8);k=torch.randn(2,4,7,8);v=torch.randn_like(k)
    alpha=torch.randn(2,4,5,7).softmax(-1)
    for method in ['QUERY','SEPARABLE']:
        g=a.ValueAdapter(method,4,8,3,chunk=2)
        with torch.no_grad():g.up.normal_()
        delta=g(q,k,v,alpha)
        projected=torch.einsum('bhsd,hdr->bhsr',v,g.down)
        phi=1+torch.tanh(torch.einsum('bhtd,hdr->bhtr',q,g.qmap))
        if method=='SEPARABLE':projected=projected*(1+torch.tanh(torch.einsum('bhsd,hdr->bhsr',k,g.kmap)))
        expected=torch.einsum('bhtr,hrd->bhtd',phi*torch.einsum('bhts,bhsr->bhtr',alpha,projected),g.up)
        assert torch.allclose(delta,expected,atol=2e-5)
        g.chunk=99
        assert torch.allclose(g(q,k,v,alpha),delta,atol=2e-5)

def test_full_vector_not_equal_query_only_in_general():
    a=api();torch.manual_seed(7)
    q=torch.randn(1,2,3,4);k=torch.randn(1,2,6,4);v=torch.randn_like(k);alpha=torch.randn(1,2,3,6).softmax(-1)
    g=a.ValueAdapter('VECTOR_PAIR',2,4,3)
    with torch.no_grad():g.up.normal_()
    before=g(q,k,v,alpha)
    changed=k.clone();changed[:,:,1]*=9
    assert not torch.allclose(before,g(q,changed,v,alpha))

def test_padding_mask_is_preserved():
    a=api();model=tiny();h=a.attach(model,'VECTOR_PAIR',[0],rank=2)
    with torch.no_grad():
        for p in h.parameters():p.add_(torch.randn_like(p)*.02)
        x=torch.tensor([[9,8,7,6]]);mask=torch.tensor([[0,1,1,1]])
        y=model(x,attention_mask=mask).logits
        x[0,0]=4;z=model(x,attention_mask=mask).logits
        assert torch.allclose(y[:,1:],z[:,1:],atol=2e-6,rtol=2e-5)
    h.remove()

def test_half_precision_backbone_keeps_adapter_gradients_float32():
    a=api();model=tiny().half();h=a.attach(model,'VECTOR_PAIR',[0],rank=2,chunk=2)
    x=torch.tensor([[4,6,7,8]])
    loss=model(x,labels=x,use_cache=False).loss;loss.backward()
    assert torch.isfinite(loss)
    assert all(p.dtype==torch.float32 for p in h.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters())
    h.remove()

def tiny_phi3():
    from transformers import Phi3Config,Phi3ForCausalLM
    torch.manual_seed(7)
    c=Phi3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                 max_position_embeddings=128,original_max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2,
                 attention_dropout=0.0,resid_pdrop=0.0,embd_pdrop=0.0,tie_word_embeddings=False)
    model=Phi3ForCausalLM(c).eval();model.set_attn_implementation('eager');return model

@pytest.mark.parametrize('method',['QUERY','SEPARABLE','VECTOR_PAIR','SCALAR_PAIR','LORA_QKVO','LORA_O'])
def test_phi3_identity_gradient_restore(method):
    a=api();model=tiny_phi3();x=torch.tensor([[3,4,5,6,7]])
    with torch.no_grad():native=model(x).logits
    h=a.attach(model,method,[0,1],rank=3,chunk=2)
    assert torch.allclose(model(x).logits,native,atol=1e-6,rtol=1e-5)
    model(x,labels=x,use_cache=False).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters())
    if method=='LORA_QKVO':assert {k.split('_',1)[1] for k in h.modules}=={'qkv_proj','o_proj'}
    with torch.no_grad():
        for p in h.parameters():p.add_(torch.randn_like(p)*.05)
        assert not torch.allclose(model(x).logits,native,atol=1e-5)
    h.remove()
    with torch.no_grad():assert torch.equal(model(x).logits,native)

def test_phi3_rejects_unfused_lora_and_cache_matches_full():
    a=api();model=tiny_phi3()
    with pytest.raises(ValueError):a.attach(model,'LORA_V',[0])
    assert getattr(model,'_memlearn_handle',None) is None
    h=a.attach(model,'VECTOR_PAIR',[0,1],rank=3,chunk=2)
    with torch.no_grad():
        for p in h.parameters():p.add_(torch.randn_like(p)*.04)
        x=torch.tensor([[4,8,9,2,5]]);full=model(x,use_cache=False).logits
        prefix=model(x[:,:3],use_cache=True)
        suffix=model(x[:,3:],past_key_values=prefix.past_key_values,use_cache=True).logits
        assert torch.allclose(full[:,3:],suffix,atol=2e-6,rtol=2e-5)
    h.remove()

def test_lora_rank_matching_unchanged_for_qwen3():
    from memlearn_attention import runner
    model=tiny();c=model.config;h=c.num_attention_heads;d=c.head_dim;hidden=c.hidden_size;kv=c.num_key_value_heads*d
    cfg=dict(method_ranks={},match_parameter_budget=True,rank=8)
    old={'LORA_V':hidden+kv,'LORA_O':h*d+hidden,'LORA_QK':2*hidden+h*d+kv,'LORA_QKVO':4*hidden+2*h*d+2*kv}
    for m,f in old.items():assert runner.rank_for(model,m,cfg)==min(128,max(1,round(4*h*d*8/f)))
