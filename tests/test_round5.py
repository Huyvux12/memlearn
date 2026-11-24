import pytest
import torch
from memlearn_attention import data_v2,mech_v2,mech_e4,mech_path,probe_v2,runner

def tiny(kind='qwen3',layers=3,init=0.02):
    torch.manual_seed(5)
    common=dict(initializer_range=init,vocab_size=258,hidden_size=24,intermediate_size=48,num_hidden_layers=layers,num_attention_heads=4,
                num_key_value_heads=2,max_position_embeddings=8192,attention_dropout=0.)
    if kind=='qwen3':
        from transformers import Qwen3Config,Qwen3ForCausalLM;m=Qwen3ForCausalLM(Qwen3Config(head_dim=6,**common))
    elif kind=='phi3':
        from transformers import Phi3Config,Phi3ForCausalLM;m=Phi3ForCausalLM(Phi3Config(pad_token_id=0,**common))
    else:
        from transformers import Gemma2Config,Gemma2ForCausalLM;m=Gemma2ForCausalLM(Gemma2Config(head_dim=6,query_pre_attn_scalar=6,sliding_window=4096,pad_token_id=0,**common))
    m.set_attn_implementation('eager');return m.eval(),runner.TestByteTokenizer()

def setup(kind='qwen3',init=0.02):
    model,tok=tiny(kind,init=init);item=mech_e4.make_items(dict(mech_e4.preset('e4_smoke'),n_test=1,n_ood=0))[0]
    lay=mech_e4.layout(tok,item);ids=mech_e4._ids(tok,item,lay,'cpu')
    return model,tok,item,lay,ids,mech_v2.family_module(model)

def full_block(lay,family):return mech_v2.blocked(lay['later'],lay['earlier'],lay['start'],'cpu',module=family)

@pytest.mark.parametrize('kind',['qwen3','phi3'])
def test_dec_all_rebuilds_the_clean_residual_after_the_path_layer(kind):
    model,tok,item,lay,ids,family=setup(kind);L=1
    clean=mech_path.components(model,ids,lay['later'],L)
    with full_block(lay,family):blocked=mech_path.components(model,ids,lay['later'],L)
    target=mech_e4.clean_states(model,ids,lay['later'])[L]
    hook=mech_path.add_hook(model.model.layers[L],lay['later'],mech_path.delta(model,clean,blocked,'all',mlp=True),lay['start'])
    try:
        with full_block(lay,family):patched=mech_e4.clean_states(model,ids,lay['later'])[L]
    finally:hook.remove()
    with full_block(lay,family):unpatched=mech_e4.clean_states(model,ids,lay['later'])[L]
    err=(patched-target).abs().max();gap=(unpatched-target).abs().max()
    assert gap>0 and err<1e-3*gap,(err,gap)

def test_delta_is_additive_over_heads_and_mlps():
    model,tok,item,lay,ids,family=setup();L=2
    clean=mech_path.components(model,ids,lay['later'],L)
    with full_block(lay,family):blocked=mech_path.components(model,ids,lay['later'],L)
    every={l:list(range(4)) for l in range(L+1)};half={l:[0,1] for l in range(L+1)};rest={l:[2,3] for l in range(L+1)}
    d=lambda h,m=False:mech_path.delta(model,clean,blocked,h,mlp=m)
    assert torch.allclose(d('all',True),d('all')+d(None,True),atol=1e-5)
    assert torch.allclose(d(every),d('all'),atol=1e-5) and torch.allclose(d(half)+d(rest),d('all'),atol=1e-5)
    assert d(None) is None

@pytest.mark.parametrize('kind',['qwen3','phi3'])
def test_restoring_every_key_and_value_of_the_later_source_restores_the_answer(kind):
    # Positions outside the later source are never blocked, so with clean K/V at the later positions they compute as in the clean run.
    model,tok,item,lay,ids,family=setup(kind);n=len(model.model.layers)
    clean=probe_v2.score_candidates(model,tok,item,'cpu',8192)['logprobs']
    with full_block(lay,family):blocked=probe_v2.score_candidates(model,tok,item,'cpu',8192)['logprobs']
    groups=mech_path.kv_groups(model,[(l,h) for l in range(n) for h in range(4)]);assert groups=={l:[0,1] for l in range(n)}
    for which,should_match in (('kv',True),('k',False)):
        targets=mech_path.kv_targets(model,groups,which)
        hooks=mech_path.replace_hooks(targets,mech_path.capture(model,ids,targets,lay['later']),lay['later'],lay['start'])
        try:
            with full_block(lay,family):got=probe_v2.score_candidates(model,tok,item,'cpu',8192)['logprobs']
        finally:
            for h in hooks:h.remove()
        assert torch.allclose(torch.tensor(got),torch.tensor(clean),atol=1e-4)==should_match
    assert not torch.allclose(torch.tensor(blocked),torch.tensor(clean),atol=1e-4)

def test_kv_groups_map_query_heads_to_shared_groups():
    model,*_=setup()
    assert mech_path.kv_groups(model,[(0,0),(0,1),(0,3),(2,2)])=={0:[0,1],2:[1]}

def test_gemma_kernel_receives_scaling_by_keyword():
    # Gemma orders (attention_mask, dropout, scaling); a positional call would silently swap them.
    model,tok,item,lay,ids,family=setup('gemma2')
    with torch.no_grad():ref=model(input_ids=ids).logits
    with mech_v2.blocked([],[],lay['start'],'cpu',module=family) as st:
        with torch.no_grad():same=model(input_ids=ids).logits
        assert st['applied']==len(model.model.layers)
    with full_block(lay,family):
        with torch.no_grad():diff=model(input_ids=ids).logits
    assert torch.allclose(ref,same,atol=1e-5) and not torch.allclose(ref,diff,atol=1e-3)

def test_prompt_text_moves_the_contract_into_the_user_turn_only_when_the_template_rejects_system():
    class NoSystem(runner.TestByteTokenizer):
        def apply_chat_template(self,msgs,**kw):
            if any(m['role']=='system' for m in msgs):raise ValueError('System role not supported')
            return super().apply_chat_template(msgs,**kw)
    item=mech_e4.make_items(dict(mech_e4.preset('e4_smoke'),n_test=1,n_ood=0))[0];msgs=data_v2.messages(item)
    text=probe_v2.prompt_text(NoSystem(),item)
    assert text.index(msgs[0]['content'])<text.index(msgs[1]['content'])
    assert probe_v2.prompt_text(runner.TestByteTokenizer(),item)!=text

def test_path_smoke_end_to_end(tmp_path):
    cfg=dict(mech_path.preset('path_smoke'),out=str(tmp_path/'p'))
    result=mech_path.run(cfg);assert result['status']=='COMPLETE' and result['completed']==result['expected']>0,result
    store=runner.Store(cfg['out'],cfg);rows=store.load('path');variants={r['variant'] for r in rows}
    assert {'clean','blocked','blocked_vals','dec_all','dec_attn','dec_mlp','dec_top1','dec_rand1_0','dec_top1_mlp',
            'kv_read1','k_read1','v_read1','kv_rand1_0','bind_top1','bind_rand1_0'}<=variants
    assert {r['variant'] for r in store.load('pathmass')}>={'clean','blocked','blocked_vals','bind_top1'}
    assert mech_path.run(cfg)['completed']==len(rows)  # resume adds nothing

# ---------------------------------------------------------------- round 6: keys per query head
def _scores(model,tok,item,lay,family,ctx_fn):
    with full_block(lay,family):
        with ctx_fn():return torch.tensor(probe_v2.score_candidates(model,tok,item,'cpu',8192)['logprobs'])

@pytest.mark.parametrize('kind',['qwen3','phi3','gemma2'])
def test_keys_for_every_query_head_of_a_group_equal_group_key_restoration(kind):
    import contextlib
    model,tok,item,lay,ids,family=setup(kind,init=0.2);n=len(model.model.layers)  # larger weights: single heads must move the scores
    keys=mech_path.capture_keys(model,ids,family,set(range(n)),lay['later'])
    groups={l:[1] for l in range(n)};targets=mech_path.kv_targets(model,groups,'k')
    saved=mech_path.capture(model,ids,targets,lay['later'])
    @contextlib.contextmanager
    def group_k():
        hooks=mech_path.replace_hooks(targets,saved,lay['later'],lay['start'])
        try:yield
        finally:
            for h in hooks:h.remove()
    per_head=_scores(model,tok,item,lay,family,lambda:mech_path.head_keys(family,keys,{l:[2,3] for l in range(n)},lay['later'],lay['start']))
    grouped=_scores(model,tok,item,lay,family,group_k)
    one=_scores(model,tok,item,lay,family,lambda:mech_path.head_keys(family,keys,{l:[2] for l in range(n)},lay['later'],lay['start']))
    blocked=_scores(model,tok,item,lay,family,contextlib.nullcontext)
    assert torch.allclose(per_head,grouped,atol=1e-5)
    assert (one-grouped).abs().max()>0 and (one-blocked).abs().max()>0  # one head of the pair is a distinct intervention
    assert family.eager_attention_forward is not None and all(m.self_attn.num_key_value_groups==2 for m in model.model.layers)

def test_heads_part_draws_leave_earlier_random_sets_unchanged(tmp_path):
    import json
    base=dict(mech_path.preset('path_smoke'),parts=['sender','receiver','route'],out=str(tmp_path/'a'))
    full=dict(base,parts=['sender','receiver','route','heads'],out=str(tmp_path/'b'))
    mech_path.run(base);mech_path.run(full)
    a=json.loads((tmp_path/'a'/'head_scores.json').read_text())['head_sets'];b=json.loads((tmp_path/'b'/'head_scores.json').read_text())['head_sets']
    assert all(b[k]==v for k,v in a.items()) and any(k.startswith('kh_') for k in b)
    rows=runner.Store(full['out'],full).load('path')
    assert {'kh_read1','kh_read2','kh_rest1','kh_rand1_0'}<={r['variant'] for r in rows}
