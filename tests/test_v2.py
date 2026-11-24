import json
import pytest
import torch
from memlearn_attention import data_v2 as d
from memlearn_attention.data import _CUES

def test_rules_are_derangements_and_candidates_distinct():
    for i in range(200):
        b=d.make_bundle(5,'test_id',i,'direct')
        assert all(b['p'][v]!=v and b['q'][v]!=v and b['q'][v]!=b['p'][v] for v in b['codebook'].values())
        assert len(set(b['mode_keys'].values()))==len(d.MODES)
        for mode in d.MODES:
            values=[v for _,v in d.candidates(b,mode)]
            assert len(set(values))==4 and d.candidates(b,mode)[0][0]=='gold'

def test_hand_computed_candidates_and_prefix():
    b=dict(codebook={'x1':'a','x2':'b','x3':'c'},p={'a':'b','b':'c','c':'a'},q={'a':'c','b':'a','c':'b'},
           mode_keys={'READ_DEC':'x1','READ_RULE':'x1','APPLY_P':'x2','APPLY_Q':'x2','NONAPPLY':'x3'},
           other_draw={m:0. for m in d.MODES})
    with pytest.raises(ValueError):d.candidates(b,'APPLY_P')  # 3 values cannot give a distinct 4th candidate
    b['codebook']['x4']='d';b['p']['d']='a';b['q']['d']='b'
    assert d.candidates(b,'APPLY_P')==[('gold','c'),('base','b'),('q','a'),('other','d')]
    assert d.candidates(b,'NONAPPLY')==[('gold','c'),('p','a'),('q','b'),('other','d')]
    assert dict(d.candidates(b,'READ_DEC'))['input_key']=='x1'
    assert d.prefix(b,'READ_RULE')=='{"decoded_value":"a","rule_result":"'

def test_render_conditions_and_q_cue():
    b=d.make_bundle(5,'test_ood',0,'jurisdiction')
    neutral=d.render(b,'NONAPPLY',('neutral','none','book_first'))
    pressured=d.render(b,'NONAPPLY',('default','q','rule_first'))
    assert d.DEFAULT_FRAMING not in neutral and 'RULE_MAP for Q' not in neutral
    assert d.DEFAULT_FRAMING in pressured and 'RULE_MAP for Q' in pressured
    assert pressured.index('RULE_MAP for P')<pressured.index('CODEBOOK')
    for family,(yes,no) in _CUES.items():
        b=d.make_bundle(5,'test_id',1,family)
        text=d.render(b,'APPLY_Q',('neutral','q','book_first')).split('REQUEST: ')[1]
        assert 'Q' in text and ' P' not in text and not text.startswith('P ')
    assert 'APPLY_Q' not in d.modes_for(('neutral','none','book_first'))

def test_items_never_carry_gold_into_prompt():
    ds=d.dataset(1,1,seed=3)
    for item in d.items(ds,('FULL',)):
        gold=d.candidates(item['bundle'],item['mode'])[0][1]
        user=d.messages(item)[1]['content']
        assert 'gold' not in user.lower()
        if item['mode'] in ('APPLY_P','APPLY_Q','NONAPPLY','READ_RULE'):
            assert '"'+gold+'"' not in user.split('INPUT_KEY')[1]

def tiny():
    from memlearn_attention import probe_v2,runner
    cfg=probe_v2.preset('probe_smoke')
    model,tok,_=runner.model_and_tokenizer(cfg,None)
    return probe_v2,model,tok

def test_cached_candidate_scores_equal_full_forward():
    probe,model,tok=tiny()
    item=d.items(d.dataset(1,0,seed=2),('FULL',))[3]
    scored=probe.score_candidates(model,tok,item,'cpu',8192)
    _,seqs,start=probe.sequences(tok,item)
    for s,got in zip(seqs,scored['logprobs']):
        with torch.no_grad():lp=model(input_ids=torch.tensor([s])).logits[0].float().log_softmax(-1)
        want=float(sum(lp[j-1,s[j]] for j in range(start,len(s))))
        assert abs(got-want)<1e-3

def test_decide_marks_leakage_only_for_nonapply_p_choice():
    probe,_,_=tiny()
    scored=dict(labels=['gold','p','q','other'],values=['a','b','c','d'],logprobs=[-3.,-1.,-2.,-4.])
    assert probe.decide(dict(mode='NONAPPLY'),scored)['leakage']==1
    assert probe.decide(dict(mode='APPLY_P'),scored)['leakage']==0
    out=probe.decide(dict(mode='APPLY_P'),dict(scored,labels=['gold','base','q','other']))
    assert out['correct']==0 and out['cross_rule']==0

def test_probe_smoke_end_to_end(tmp_path):
    from memlearn_attention import probe_v2
    cfg=probe_v2.preset('probe_smoke');cfg['out']=str(tmp_path)
    result=probe_v2.run(cfg)
    assert result['status']=='COMPLETE' and result['completed']==result['expected']>0
    assert (tmp_path/'probe_by_cell.csv').exists() and (tmp_path/'probe_jss_v2.csv').exists()
    decision=json.loads((tmp_path/'probe_decision.json').read_text())
    assert 'target_failure_exists' in decision
    assert probe_v2.run(cfg)['completed']==result['completed']  # resume adds nothing
