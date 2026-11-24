import importlib.util
import json
import pytest
import torch
from pathlib import Path

def api():
    assert importlib.util.find_spec('memlearn_attention.runner'), 'runner missing'
    from memlearn_attention import runner
    return runner

def test_atomic_receipts_replay_and_config_guard(tmp_path):
    r=api();s=r.Store(tmp_path,{'model':'tiny','seed':1})
    row={'id':'x','method':'FULL','seed':1,'output':'{}','correct':1}
    s.save('predictions',row)
    assert s.load('predictions')==[row]
    assert not s.save('predictions',row)
    with pytest.raises(ValueError):s.save('predictions',dict(row,output='changed'))
    with pytest.raises(ValueError):r.Store(tmp_path,{'model':'different'})
    f=next((tmp_path/'predictions').glob('*.json'));obj=json.loads(f.read_text());obj['row']['correct']=0;f.write_text(json.dumps(obj))
    with pytest.raises(ValueError):s.load('predictions')

def test_balanced_schedule_pairs_methods_before_new_bundle():
    r=api();from memlearn_attention.data import dataset
    ds=dataset(1,1,2,2,seed=7)
    schedule=r.schedule(ds,['FULL','QUERY'],[7])
    first=schedule[:8]
    assert len({x['request']['bundle_id'] for x in first})==1
    assert {x['method'] for x in first}=={'FULL','QUERY'}
    assert all(len([x for x in first if x['method']==m])==4 for m in ['FULL','QUERY'])

def test_tokenization_masks_prompt_and_never_truncates_silently():
    r=api();from memlearn_attention.data import dataset,requests_for
    tok=r.TestByteTokenizer();req=requests_for(dataset(1,1,1,1)['train'][0])[0]
    batch=r.training_example(tok,req,4096,'cpu')
    prompt_n=(batch['labels']==-100).sum().item()
    assert prompt_n>0 and (batch['labels']!=-100).sum()>0
    assert torch.equal(batch['input_ids'][:,prompt_n:],batch['labels'][:,prompt_n:])
    with pytest.raises(ValueError):r.training_example(tok,req,10,'cpu')

def test_tiny_actual_training_and_report_preserve_negative_results(tmp_path):
    r=api()
    config=r.preset('smoke');config.update(out=str(tmp_path),budget_seconds=180)
    result=r.run(config)
    assert result['execution_kind']=='RANDOM_QWEN_TEST_ONLY'
    rows=json.loads((tmp_path/'audit.json').read_text())
    assert rows['receipts_verified']>0 and not rows['scientific_pretrained_results']
    assert (tmp_path/'report.md').exists() and (tmp_path/'metrics_by_method.csv').exists()
    logs=r.Store(tmp_path,config).load('training')
    assert logs and any(x['adapter_changed'] for x in logs)
    # Same exact config resume creates no duplicates or repeated optimizer steps.
    before=len(list((tmp_path/'predictions').glob('*.json')))
    r.run(config)
    assert len(list((tmp_path/'predictions').glob('*.json')))==before

def test_export_marks_partial_without_blanket_invalid(tmp_path):
    r=api();from memlearn_attention.data import dataset,requests_for
    from memlearn_attention.metrics import score
    cfg=r.preset('smoke');cfg['out']=str(tmp_path)
    s=r.Store(tmp_path,cfg);req=requests_for(dataset(1,1,1,1)['test_id'][0])[0]
    row=dict(id=req['id'],method='FULL',seed=req['seed'],split=req['split'],bundle_id=req['bundle_id'],mode=req['mode'],output=json.dumps(req['gold']),**score(req,json.dumps(req['gold'])))
    s.save('predictions',row);r.export(s,cfg,algebra={})
    q=json.loads((tmp_path/'validity_by_rq.json').read_text())
    assert q['RQ2']['evidence']['full_accuracy']==1
    assert q['RQ3']['coverage']=='NOT_RUN'

def test_reliability_requires_repeats_of_same_instance():
    r=api();rows=[]
    for repeat,c in [(0,1),(1,0),(2,1)]:
        rows.append(dict(method='A',seed=1,split='test_id',bundle_id='x',mode='READ',repeat=repeat,correct=c))
    result=r.reliability(rows,k=3)
    assert result['pass_at_k_empirical']==1 and result['pass_power_k_empirical']==0
    assert r.reliability(rows[:1],k=3)['eligible_instances']==0

def test_matched_control_keeps_exact_input_token_count():
    r=api();from memlearn_attention.data import dataset,requests_for
    tok=r.TestByteTokenizer();req=requests_for(dataset(1,1,1,1)['test_id'][0])[2]
    normal=r.prompt_tokens(tok,req,'FULL')
    matched=r.prompt_tokens(tok,req,'FACTS_MATCHED')
    alternate=r.prompt_tokens(tok,req,'FACTS_MATCHED_ALT')
    assert len(normal)==len(matched)==len(alternate)
    assert normal!=matched and matched!=alternate
    text=tok.decode(matched)
    assert 'INPUT_KEY' in text and req['bundle']['key'] in text
    assert 'RULE_MAP for P:' not in text

def test_checkpoint_roundtrip_after_optimizer_update(tmp_path):
    r=api();from test_attention import tiny
    from memlearn_attention.attention import attach
    model=tiny();h=attach(model,'VECTOR_PAIR',[0],2)
    x=torch.tensor([[4,8,6]])
    optimizer=torch.optim.AdamW(h.parameters(),lr=.01)
    model(x,labels=x,use_cache=False).loss.backward();optimizer.step()
    expected=model(x).logits.detach();cfg={'a':1};s=r.Store(tmp_path,cfg)
    path=r.checkpoint_path(s,'VECTOR_PAIR',1)
    r.save_checkpoint(path,{'adapter':h.state_dict(),'optimizer':optimizer.state_dict(),'config_sha256':s.fingerprint})
    h.remove();h=attach(model,'VECTOR_PAIR',[0],2)
    h.load_state_dict(r.read_checkpoint(path,s)['adapter'])
    assert torch.equal(expected,model(x).logits.detach());h.remove()

def test_invalid_configuration_rejected_before_model_load(tmp_path):
    r=api();cfg=r.preset('smoke');cfg['out']=str(tmp_path);cfg['methods']=['FULL','FULL']
    with pytest.raises(ValueError,match='distinct'):r.run(cfg)

def test_export_includes_condition_cost_and_decision_tables(tmp_path):
    r=api();cfg=r.preset('smoke');cfg['out']=str(tmp_path)
    r.run(cfg)
    assert (tmp_path/'metrics_by_condition.csv').exists()
    assert (tmp_path/'training_cost.csv').exists()
    report=(tmp_path/'report.md').read_text()
    assert '\n\n|---' not in report

def test_gradient_checkpointing_training_is_real(tmp_path):
    r=api();cfg=r.preset('smoke');cfg.update(out=str(tmp_path),methods=['VECTOR_PAIR'],gradient_checkpointing=True)
    value=r.run(cfg)
    assert value['status']=='COMPLETE'
    rows=r.Store(tmp_path,cfg).load('training')
    assert len(rows)==2 and all(x['adapter_changed'] for x in rows)

def test_training_cost_first_last_follow_optimizer_step_not_hash_order(tmp_path):
    import csv
    r=api();cfg=r.preset('smoke');cfg['out']=str(tmp_path);store=r.Store(tmp_path,cfg)
    for step in range(1,33):
        store.save('training',dict(id=f'step_{step}',method='QUERY',seed=17,step=step,loss=float(33-step),trainable_params=100,latency_s=.01,input_tokens=5,peak_vram_bytes=None))
    r.export(store,cfg)
    row=list(csv.DictReader((tmp_path/'training_cost.csv').open()))[0]
    assert float(row['first_loss'])==32 and float(row['last_loss'])==1

def test_resume_optimizer_error_removes_adapter_before_baseline(tmp_path):
    import time
    r=api();from test_attention import tiny
    from memlearn_attention.attention import attach
    from memlearn_attention.data import dataset
    cfg=r.preset('smoke');cfg['out']=str(tmp_path);s=r.Store(tmp_path,cfg);model=tiny()
    x=torch.tensor([[4,5,6]])
    with torch.no_grad():expected=model(x).logits
    rank=r.rank_for(model,'QUERY',cfg);h=attach(model,'QUERY',[0],rank)
    with torch.no_grad():
        for p in h.parameters():p.add_(torch.randn_like(p)*.2)
    r.save_checkpoint(r.checkpoint_path(s,'QUERY',17),dict(adapter=h.state_dict(),optimizer={},step=1,total_steps=2,config_sha256=s.fingerprint))
    h.remove()
    with pytest.raises(KeyError):r.train_method(model,r.TestByteTokenizer(),dataset(1,1,1,1),'QUERY',17,cfg,s,time.monotonic()+20)
    assert getattr(model,'_memlearn_handle',None) is None
    with torch.no_grad():assert torch.equal(expected,model(x).logits)

def test_reliability_reports_each_method_separately():
    r=api();rows=[]
    for method,correct in [('A',1),('B',0)]:
        for repeat in range(3):
            rows.append(dict(method=method,seed=1,split='test_id',bundle_id='x',mode='READ',repeat=repeat,correct=correct))
    result=r.reliability(rows,k=3)
    panels={p['method']:p for p in result['by_method']}
    assert panels['A']['pass_power_k_empirical']==1 and panels['B']['pass_power_k_empirical']==0
