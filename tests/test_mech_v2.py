import json
import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3 as native
from memlearn_attention import data_v2,mech_v2,probe_v2,runner

def tiny():
    cfg=mech_v2.preset('mech_smoke');model,tok,_=runner.model_and_tokenizer(cfg,None);return model,tok

def item(order,mode='APPLY_P'):
    b=data_v2.make_bundle(3,'test_id',0,'direct')
    return dict(bundle=b,bundle_id=b['id'],split='test_id',family='direct',mode=mode,condition=('neutral','none',order),control='FULL')

@pytest.mark.parametrize('order',['book_first','rule_first','rule_first_bookrepeat'])
def test_spans_are_disjoint_nonempty_and_sham_is_size_matched(order):
    _,tok=tiny();it=item(order);cands,seqs,start=probe_v2.sequences(tok,it)
    head=probe_v2.prompt_text(tok,it)+data_v2.prefix(it['bundle'],it['mode'])+cands[0][1]+'"}'
    s=mech_v2.token_spans(tok,head,start)
    assert s['ids']==seqs[0][:start] and s['book'] and s['rule'] and not set(s['book'])&set(s['rule'])
    assert len(s['sham_for_book'])==len(s['book']) and len(s['sham_for_rule'])==len(s['rule'])
    assert 0 not in s['sham_for_book'] and not set(s['sham_for_book'])&(set(s['book'])|set(s['rule']))
    if order=='book_first':assert max(s['book'])<min(s['rule'])
    else:assert max(s['rule'])<min(s['book'])

def test_block_changes_only_blocked_queries_and_later_tokens_and_restores():
    model,tok=tiny();it=item('book_first');cands,seqs,start=probe_v2.sequences(tok,it)
    head=probe_v2.prompt_text(tok,it)+data_v2.prefix(it['bundle'],it['mode'])+cands[0][1]+'"}'
    s=mech_v2.token_spans(tok,head,start);ids=torch.tensor([seqs[0][:start]]);before=native.eager_attention_forward
    with torch.no_grad():
        base=model(input_ids=ids,output_hidden_states=True).hidden_states[-1]
        with mech_v2.blocked(s['rule'],s['book'],start,'cpu') as state:
            cut=model(input_ids=ids,output_hidden_states=True).hidden_states[-1]
            assert state['applied']==len(model.model.layers)
    first=min(s['rule'])
    assert torch.allclose(base[0,:first],cut[0,:first],atol=1e-6)
    assert not torch.allclose(base[0,first],cut[0,first],atol=1e-6)
    assert native.eager_attention_forward is before

def test_block_refuses_to_stack_on_another_patch():
    before=native.eager_attention_forward;native.eager_attention_forward=lambda *a,**k:None
    try:
        with pytest.raises(RuntimeError):
            with mech_v2.blocked([1],[0],4,'cpu'):pass
    finally:native.eager_attention_forward=before

def test_mech_smoke_end_to_end(tmp_path):
    cfg=mech_v2.preset('mech_smoke');cfg['out']=str(tmp_path)
    r=mech_v2.run(cfg)
    assert r['status']=='COMPLETE' and r['completed']==r['expected']>0
    rows=[json.loads(p.read_text())['row'] for p in (tmp_path/'mech').glob('*.json')]
    assert {x['variant'] for x in rows}=={v[0] for v in mech_v2.VARIANTS if v[4] is None}
    assert all(x['blocked_pairs']>0 for x in rows if x['blocked_query'])
    assert (tmp_path/'mech_by_variant.csv').exists()

def test_band_layers_partition_36_and_3_layers():
    assert mech_v2.band_layers('early',36)==list(range(12)) and mech_v2.band_layers('mid',36)==list(range(12,24))
    assert mech_v2.band_layers('late',36)==list(range(24,36)) and mech_v2.band_layers(None,4)==[0,1,2,3]
    assert [mech_v2.band_layers(b,3) for b in ('early','mid','late')]==[[0],[1],[2]]

def test_codebook_key_and_value_spans_partition_entries():
    _,tok=tiny();it=item('book_first');cands,seqs,start=probe_v2.sequences(tok,it)
    s=mech_v2.token_spans(tok,probe_v2.prompt_text(tok,it)+data_v2.prefix(it['bundle'],it['mode'])+cands[0][1]+'"}',start)
    keys,values=set(s['book_keys']),set(s['book_values'])
    assert keys and values and not keys&values and keys|values<=set(s['book'])
    decoded=''.join(tok.decode([s['ids'][i]]) for i in s['book_keys'])
    assert all(k in decoded for k in it['bundle']['codebook'])

def test_band_block_touches_only_band_layers():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(0)
    c=Qwen3Config(vocab_size=258,hidden_size=24,intermediate_size=48,num_hidden_layers=3,num_attention_heads=3,
                  num_key_value_heads=1,head_dim=8,max_position_embeddings=512,attention_dropout=0.)
    model=Qwen3ForCausalLM(c);model.set_attn_implementation('eager');model.eval()
    assert mech_v2.family_module(model) is native
    ids=torch.randint(2,250,(1,20))
    with torch.no_grad():
        base=model(input_ids=ids,output_hidden_states=True).hidden_states
        with mech_v2.blocked(list(range(10,20)),list(range(1,9)),20,'cpu',layers=[1],module=native) as state:
            cut=model(input_ids=ids,output_hidden_states=True).hidden_states
            assert state['applied']==1
    assert torch.allclose(base[1],cut[1],atol=1e-6)          # output of layer 0: untouched
    assert not torch.allclose(base[2][0,10:],cut[2][0,10:])   # layer 1 blocked rows change
    assert torch.allclose(base[2][0,:10],cut[2][0,:10],atol=1e-6)

def test_e2_variants_end_to_end_on_tiny(tmp_path):
    cfg=mech_v2.preset('mech_smoke');cfg.update(out=str(tmp_path),variants=['M0','M1','M2','N6','N7'])
    r=mech_v2.run(cfg)
    assert r['status']=='COMPLETE' and r['completed']==r['expected']>0
    rows=[json.loads(p.read_text())['row'] for p in (tmp_path/'mech').glob('*.json')]
    assert {x['variant'] for x in rows}=={'M0','M1','M2','N6','N7'}
    assert all(x['blocked_pairs']>0 for x in rows if x['variant'] in ('N6','N7'))
