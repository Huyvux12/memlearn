import importlib.util
import json
import pytest

def modules():
    assert importlib.util.find_spec('memlearn_attention.data'), 'generator not implemented'
    from memlearn_attention import data, metrics
    return data, metrics

def test_oracle_modes_and_derangement():
    d, m = modules()
    b = d.make_bundle(13, 'train', 0, 'direct')
    assert all(k != v for k, v in b['rule'].items())
    reqs = d.requests_for(b)
    assert {r['mode'] for r in reqs} == {'READ','APPLY','NONAPPLY','MIXED'}
    for r in reqs:
        assert m.score(r, json.dumps(r['gold']))['correct'] == 1
        assert r['id'] not in d.render(r)
    read, apply, plain, mixed = reqs
    assert apply['gold']['answer'] != plain['gold']['answer']
    assert mixed['gold']['read'] == read['gold']
    assert mixed['gold']['apply'] == apply['gold']

def test_split_ids_and_codebooks_are_disjoint():
    d, _ = modules()
    a=d.dataset(4,4,4,2,seed=5)
    assert len({b['id'] for bs in a.values() for b in bs}) == 18
    assert not set(b['family'] for b in a['train']) & set(b['family'] for b in a['test_ood'])
    assert a==d.dataset(4,4,4,2,seed=5)

def test_strict_parse_no_answer_salvage_and_leakage_fingerprint():
    d,m=modules(); r=d.requests_for(d.make_bundle(3,'test_id',0,'direct'))[2]
    assert m.score(r,'reasoning '+json.dumps(r['gold']))['parse_error']==1
    assert m.score(r,json.dumps({'answer':r['rule_answer']}))['rule_leakage']==1
    assert m.score(r,json.dumps({'answer':'other'}))['rule_leakage']==0
    assert m.score(r,'{"answer":')['correct']==0

def test_jss_is_bundle_product_not_product_of_means():
    _,m=modules()
    rows=[]
    for b, outcomes in [('a',[1,1,0]),('b',[0,1,1])]:
        for mode,c in zip(['READ','APPLY','NONAPPLY'],outcomes):
            rows.append(dict(method='FULL',seed=1,split='test_id',bundle_id=b,mode=mode,correct=c,rule_leakage=0,parse_error=0))
    panel=m.summarize(rows)[0]
    assert panel['jss']==0 and panel['jss_n']==2

def test_unpaired_partial_rows_never_become_negative_transfer():
    _,m=modules()
    rows=[dict(method='A',seed=1,split='test_id',bundle_id='x',mode='READ',correct=1)]
    c=m.contrast(rows,'A','B',bootstrap=50)
    assert c['paired_bundles']==0 and c['delta_accuracy'] is None

def test_cluster_bootstrap_exact_zero_when_identical():
    _,m=modules();rows=[]
    for method in ['A','B']:
        for b in range(5):
            for mode in ['READ','APPLY','NONAPPLY','MIXED']:
                rows.append(dict(method=method,seed=1,split='test_id',bundle_id=str(b),mode=mode,correct=b%2))
    c=m.contrast(rows,'A','B',bootstrap=100)
    assert c['delta_accuracy']==0 and c['ci95']==[0,0]

def test_rq_missing_method_does_not_invalidate_measured_full():
    _,m=modules()
    rows=[dict(method='FULL',seed=1,split='test_id',bundle_id=str(i),mode=mode,correct=1,rule_leakage=0,parse_error=0) for i in range(3) for mode in ['READ','APPLY','NONAPPLY','MIXED']]
    rq=m.research_questions(rows,{},expected_bundles=3)
    assert rq['RQ2']['technical_validity'] is True
    assert rq['RQ2']['evidence']['full_accuracy']==1
    assert rq['RQ3']['coverage']=='NOT_RUN'

def test_repeated_trials_do_not_overwrite_jss_panels():
    _,m=modules();rows=[]
    for repeat,correct in [(0,1),(1,0)]:
        for mode in ['READ','APPLY','NONAPPLY','MIXED']:
            rows.append(dict(method='FULL',seed=1,split='test_id',bundle_id='a',repeat=repeat,mode=mode,correct=correct))
    p=m.summarize(rows)[0]
    assert p['jss']==.5 and p['jss_n']==2

def test_jss_paired_contrast_uses_complete_bundles_only():
    _,m=modules();rows=[]
    for method,correct in [('A',1),('B',0)]:
        for mode in ['READ','APPLY','NONAPPLY']:
            rows.append(dict(method=method,seed=1,split='test_id',bundle_id='a',mode=mode,correct=correct))
    rows.append(dict(method='A',seed=1,split='test_id',bundle_id='partial',mode='READ',correct=1))
    result=m.jss_contrast(rows,'A','B',bootstrap=50)
    assert result['paired_bundles']==1 and result['delta_jss']==1

def test_duplicate_json_keys_are_not_accepted():
    d,m=modules();r=d.requests_for(d.make_bundle(3,'test_id',0,'direct'))[2]
    text='{"answer":"wrong","answer":'+json.dumps(r['gold']['answer'])+'}'
    assert m.score(r,text)['correct']==0

def test_hand_computed_oracle_has_expected_values():
    d,_=modules()
    b=dict(codebook={'x':'red','y':'blue'},rule={'red':'blue','blue':'red'},key='x')
    assert d.oracle(b,'READ')=={'decoded_value':'red','rule_result':'blue'}
    assert d.oracle(b,'APPLY')=={'answer':'blue'}
    assert d.oracle(b,'NONAPPLY')=={'answer':'red'}

def test_rq4_candidate_seed_coverage_not_borrowed_from_baselines():
    _,m=modules();rows=[]
    for method,seeds in [('FULL',[1,2,3]),('VECTOR_PAIR',[1])]:
        for seed in seeds:
            for mode in ['READ','APPLY','NONAPPLY','MIXED']:
                rows.append(dict(method=method,seed=seed,split='test_ood',bundle_id='x',mode=mode,correct=1,rule_leakage=0))
    q=m.research_questions(rows,{})
    assert q['RQ4']['coverage']=='PARTIAL'
    assert q['RQ4']['evidence']['seeds']==[1]
