import json
import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3 as native
from memlearn_attention import data_v2,mech_v2,mech_e4,probe_v2,runner

def tiny(layers=2):
    if layers==2:
        model,tok,_=runner.model_and_tokenizer(probe_v2.preset('probe_smoke'),None);return model,tok
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(3)
    c=Qwen3Config(vocab_size=258,hidden_size=24,intermediate_size=48,num_hidden_layers=layers,num_attention_heads=3,
                  num_key_value_heads=1,head_dim=8,max_position_embeddings=8192,attention_dropout=0.)
    m=Qwen3ForCausalLM(c);m.set_attn_implementation('eager');return m.eval(),runner.TestByteTokenizer()

def e4_item():
    return mech_e4.make_items(dict(mech_e4.preset('e4_smoke'),n_test=1,n_ood=0))[0]

# ---------------------------------------------------------------- natural-language variant and framing control
def test_nl_bundle_renders_prose_and_keeps_candidate_contract():
    b=data_v2.make_bundle(5,'test_id',0,'direct',style='nl')
    assert b['style']=='nl' and all(v in data_v2.ROOMS for v in b['codebook'].values()) and all(k.isdigit() for k in b['codebook'])
    text=data_v2.render(b,'APPLY_P',('neutral','q','book_first'))
    assert 'STAFF DIRECTORY:' in text and 'PROCEDURE P (relocation):' in text and 'PROCEDURE Q (renovation):' in text and 'BADGE:' in text
    assert 'CODEBOOK' not in text and 'decoded value' not in text
    for mode in data_v2.MODES:assert len({v for _,v in data_v2.candidates(b,mode)})==4
    item=dict(bundle=b,mode='READ_DEC',condition=('neutral','none','book_first'),control='FULL')
    assert data_v2.messages(item)[0]['content']==data_v2.NL_CONTRACT
    assert data_v2.make_bundle(5,'test_id',0,'direct')['id']!=b['id']  # styles never share bundle ids

def test_codes_bundles_unchanged_by_style_parameter():
    assert data_v2.make_bundle(17,'test_id',3,'report')==data_v2.make_bundle(17,'test_id',3,'report',style='codes')
    assert data_v2.make_bundle(17,'test_id',3,'report')['id']==data_v2.digest(['v2',17,'test_id',3,'report'])[:20]

def test_filler_framing_same_slot_no_instruction():
    b=data_v2.make_bundle(5,'test_id',0,'direct')
    text=data_v2.render(b,'NONAPPLY',('filler','none','book_first'))
    assert data_v2.FILLER_FRAMING in text and data_v2.DEFAULT_FRAMING not in text
    assert text.index(data_v2.FILLER_FRAMING)<text.index('RULE_MAP for P')
    assert ' P' not in data_v2.FILLER_FRAMING and 'procedure' not in data_v2.FILLER_FRAMING.lower()
    with pytest.raises(ValueError):data_v2.render(b,'NONAPPLY',('loud','none','book_first'))

@pytest.mark.parametrize('style',['codes','nl'])
def test_mech_spans_work_for_both_styles(style):
    _,tok=tiny();b=data_v2.make_bundle(3,'test_id',0,'direct',style=style)
    it=dict(bundle=b,bundle_id=b['id'],split='test_id',family='direct',mode='APPLY_P',condition=('neutral','none','book_first'),control='FULL')
    cands,seqs,start=probe_v2.sequences(tok,it)
    s=mech_v2.token_spans(tok,probe_v2.prompt_text(tok,it)+data_v2.prefix(b,'APPLY_P')+cands[0][1]+'"}',start)
    assert s['book'] and s['rule'] and s['book_keys'] and s['book_values'] and not set(s['book'])&set(s['rule'])
    assert set(s['book_keys'])|set(s['book_values'])<=set(s['book'])

# ---------------------------------------------------------------- free-generation check and configurable conditions
def test_probe_generation_check_and_custom_conditions(tmp_path):
    cfg=probe_v2.preset('probe_smoke')
    cfg.update(out=str(tmp_path),style='nl',controls=['FULL'],generate_every=1,
               conditions=[['filler','none','book_first'],['neutral','q','rule_first']],modes=['APPLY_P','NONAPPLY'])
    r=probe_v2.run(cfg)
    assert r['status']=='COMPLETE' and r['completed']==r['expected']==2*2*2-0
    rows=[json.loads(p.read_text())['row'] for p in (tmp_path/'probe').glob('*.json')]
    assert {tuple(x['condition']) for x in rows}=={('filler','none','book_first'),('neutral','q','rule_first')}
    labels={'gold','p','q','other','base','input_key','invalid'}
    assert all(x['gen_choice'] in labels and x['gen_agree'] in (0,1) for x in rows)

# ---------------------------------------------------------------- E4 machinery
def test_per_head_block_on_all_heads_equals_full_block():
    model,tok=tiny(3);ids=torch.randint(2,250,(1,24));q=list(range(12,24));k=list(range(1,10))
    with torch.no_grad():
        with mech_v2.blocked(q,k,24,'cpu',module=native):full=model(input_ids=ids).logits
        heads={l:list(range(3)) for l in range(3)}
        with mech_v2.blocked(q,k,24,'cpu',module=native,heads=heads,n_heads=3) as st:
            per=model(input_ids=ids).logits;assert st['applied']==3
        with mech_v2.blocked(q,k,24,'cpu',module=native,heads={1:[0]},n_heads=3) as st:
            one=model(input_ids=ids).logits;assert st['applied']==1
        base=model(input_ids=ids).logits
    assert torch.allclose(full,per,atol=1e-6)
    assert not torch.allclose(one,base,atol=1e-6) and not torch.allclose(one,full,atol=1e-6)

def test_restore_hook_is_identity_on_clean_run_and_skips_tails():
    model,tok=tiny(3);ids=torch.randint(2,250,(1,20));pos=list(range(8,14))
    clean=mech_e4.clean_rule_states(model,ids,pos)
    with torch.no_grad():ref=model(input_ids=ids).logits
    h=mech_e4.restore_hook(model.model.layers[1],pos,clean[1],20)
    try:
        with torch.no_grad():same=model(input_ids=ids).logits;short=model(input_ids=ids[:,:10]).logits
    finally:h.remove()
    assert torch.allclose(ref,same,atol=1e-6)
    with torch.no_grad():assert torch.allclose(short,model(input_ids=ids[:,:10]).logits,atol=1e-6)

def test_restore_after_last_block_layer_recovers_downstream_rule_states():
    # With a 3-layer model, restoring clean rule states after layer 2 (the last) makes their final states clean again.
    model,tok=tiny(3);it=e4_item();lay=mech_e4.layout(tok,it);ids=torch.tensor([probe_v2.sequences(tok,it)[1][0][:lay['start']]])
    clean=mech_e4.clean_rule_states(model,ids,lay['rule'])
    with mech_v2.blocked(lay['rule'],lay['book'],lay['start'],'cpu',module=native):
        blocked=mech_e4.clean_rule_states(model,ids,lay['rule'])
        h=mech_e4.restore_hook(model.model.layers[2],lay['rule'],clean[2],lay['start'])
        try:restored=mech_e4.clean_rule_states(model,ids,lay['rule'])
        finally:h.remove()
    assert not torch.allclose(blocked[2],clean[2]) and torch.allclose(restored[2],clean[2],atol=1e-6)

def test_head_scores_shape_and_layout_maps_values():
    model,tok=tiny(3);it=e4_item();lay=mech_e4.layout(tok,it)
    assert set(lay['rule_left'])==set(lay['book_vals'])==set(it['bundle']['codebook'].values())
    s=mech_e4.head_scores(model,tok,[it],'cpu')
    assert s.shape==(3,3) and torch.isfinite(s).all()

def test_e4_smoke_end_to_end(tmp_path):
    cfg=mech_e4.preset('e4_smoke');cfg['out']=str(tmp_path)
    r=mech_e4.run(cfg)
    assert r['status']=='COMPLETE' and r['completed']==r['expected']>0
    rows=[json.loads(p.read_text())['row'] for p in (tmp_path/'e4').glob('*.json')]
    variants={x['variant'] for x in rows}
    assert {'clean','blocked','restore_L0','top1','rand1_0','rand1_1'}<=variants
    hs=json.loads((tmp_path/'head_scores.json').read_text());assert hs['head_sets']['top1']
    assert mech_e4.run(cfg)['completed']==r['completed']

def test_unequal_tail_scoring_matches_full_forward_for_nl():
    model,tok=tiny();b=data_v2.make_bundle(9,'test_id',0,'direct',style='nl')
    it=dict(bundle=b,bundle_id=b['id'],split='test_id',family='direct',mode='NONAPPLY',condition=('neutral','none','book_first'),control='FULL')
    cands,seqs,start=probe_v2.sequences(tok,it)
    assert len({len(s)-start for s in seqs})>1  # exercises the per-candidate (unequal tails) path
    scored=probe_v2.score_candidates(model,tok,it,'cpu',8192)
    for s,got in zip(seqs,scored['logprobs']):
        with torch.no_grad():lp=model(input_ids=torch.tensor([s])).logits[0].float().log_softmax(-1)
        assert abs(got-float(sum(lp[j-1,s[j]] for j in range(start,len(s)))))<1e-3

# ---------------------------------------------------------------- round 4: templates, rule-first layout, circuit pieces
def test_nl_templates_render_golden_and_new_frames():
    gold=json.load(open('tests/golden_nl_v2_0.json'))
    for g in gold:
        b=data_v2.make_bundle(41,'test_id',g['i'],['direct','report','workflow'][g['i']],style='nl')
        assert data_v2.messages(dict(bundle=b,mode=g['mode'],condition=tuple(g['cond']),control='FULL'))==g['messages']
    import re
    for style in ('nl_gate','nl_shelf'):
        b=data_v2.make_bundle(53,'test_id',0,'direct',style=style);t=data_v2.NL_TEMPLATES[style]
        text=data_v2.render(b,'APPLY_P',('neutral','q','rule_first'))
        assert t['book_head'] in text and t['rule_head']['P'] in text and t['rule_head']['Q'] in text and 'room' not in text
        pat=data_v2.span_patterns(style)
        assert len(re.findall(pat['book_entry'],re.search(pat['book'],text).group(0)))==6
        assert len(re.findall(pat['rule_entry'],re.search(pat['rule_p'],text).group(0)))==6
        assert data_v2.messages(dict(bundle=b,mode='READ_DEC',condition=('neutral','none','book_first'),control='FULL'))[0]['content']==t['contract']

@pytest.mark.parametrize('style,order',[('codes','rule_first'),('nl_gate','book_first'),('nl_shelf','rule_first')])
def test_e4_layout_tracks_later_source(style,order):
    _,tok=tiny();cfg=dict(mech_e4.preset('e4_smoke'),n_test=1,n_ood=0,style=style,order=order)
    it=mech_e4.make_items(cfg)[0];lay=mech_e4.layout(tok,it)
    bf=order=='book_first'
    assert lay['later']==(lay['rule'] if bf else lay['book']) and lay['earlier']==(lay['book'] if bf else lay['rule'])
    assert max(lay['earlier'])<min(lay['later'])
    assert set(lay['later_vals'])==set(lay['earlier_vals'])==set(it['bundle']['codebook'].values())
    assert lay['base'] in lay['rule_entry'] and set(lay['rule_entry'][lay['base']])<=set(lay['rule'])

def test_all_but_none_equals_full_block_and_reader_mass():
    model,tok=tiny(3);it=mech_e4.make_items(dict(mech_e4.preset('e4_smoke'),n_test=1,n_ood=0))[0]
    lay=mech_e4.layout(tok,it);ids=mech_e4._ids(tok,it,lay,'cpu')
    every=[(l,h) for l in range(3) for h in range(3)]
    with torch.no_grad():
        with mech_v2.blocked(lay['later'],lay['earlier'],lay['start'],'cpu',module=native):full=model(input_ids=ids).logits
        with mech_v2.blocked(lay['later'],lay['earlier'],lay['start'],'cpu',module=native,heads=mech_e4._heads(every),n_heads=3):per=model(input_ids=ids).logits
    assert torch.allclose(full,per,atol=1e-6)
    m=mech_e4.reader_mass(mech_e4._attentions(model,ids),lay)
    assert m.shape==(3,3) and torch.isfinite(m).all()

def test_e4_smoke_rule_first_nl_with_circuit_variants(tmp_path):
    cfg=mech_e4.preset('e4_smoke');cfg.update(out=str(tmp_path),style='nl_gate',order='rule_first')
    r=mech_e4.run(cfg)
    assert r['status']=='COMPLETE' and r['completed']==r['expected']>0
    rows=[json.loads(p.read_text())['row'] for p in (tmp_path/'e4').glob('*.json')]
    v={x['variant'] for x in rows}
    assert {'only_top1','read_top1','read_rand1_0','top1','rand1_1'}<=v
    path=[json.loads(p.read_text())['row'] for p in (tmp_path/'e4path').glob('*.json')]
    assert {x['variant'] for x in path}=={'clean','top1','rand1_0','rand1_1'}
    hs=json.loads((tmp_path/'head_scores.json').read_text());assert hs['kinds']['only_top1']=='bind_all_but' and hs['reader_top']
