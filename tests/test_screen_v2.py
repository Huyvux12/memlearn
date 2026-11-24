import json
import torch
from memlearn_attention import data_v2,screen_v2

def test_training_never_sees_held_out_source_order():
    cfg=screen_v2.preset('screen_v2');ds=data_v2.splits(4,1,1,seed=3)
    train=screen_v2.train_items(ds,cfg)
    assert train and all(x['condition'][2]=='book_first' for x in train)
    assert {tuple(c)[2] for c in cfg['eval_conditions']}=={'book_first','rule_first'}
    train_ids={b['id'] for b in ds['train']};test_ids={b['id'] for k in ('test_id','test_ood') for b in ds[k]}
    assert not train_ids&test_ids

def test_target_matches_probe_gold_and_prompt_is_masked():
    from memlearn_attention.runner import TestByteTokenizer
    tok=TestByteTokenizer();ds=data_v2.splits(1,1,1,seed=3)
    item=screen_v2.train_items(ds,screen_v2.preset('screen_v2'))[2]
    b=item['bundle'];gold=data_v2.candidates(b,item['mode'])[0][1]
    assert data_v2.target(b,item['mode'])==data_v2.prefix(b,item['mode'])+gold+'"}'
    ex=screen_v2.train_example(tok,item,'cpu',8192)
    n=(ex['labels']==-100).sum().item()
    assert tok.decode(ex['labels'][0,n:].tolist())==data_v2.target(b,item['mode'])
    assert torch.equal(ex['input_ids'][:,n:],ex['labels'][:,n:])

def test_screen_smoke_trains_scores_and_resumes(tmp_path):
    cfg=screen_v2.preset('screen_smoke');cfg['out']=str(tmp_path)
    result=screen_v2.run(cfg)
    assert result['status']=='COMPLETE' and result['completed']==result['expected']>0
    availability=json.loads((tmp_path/'method_availability.json').read_text())
    assert all(v['status']=='DONE' for v in availability.values())
    assert availability['FULL']['trainable_params']==0 and availability['VECTOR_PAIR']['trainable_params']>0
    assert (tmp_path/'adapter_VECTOR_PAIR.pt').exists() and (tmp_path/'screen_by_cell.csv').exists()
    rows=[json.loads(p.read_text())['row'] for p in (tmp_path/'probe').glob('*.json')]
    assert {r['method'] for r in rows}==set(cfg['methods'])
    assert screen_v2.run(cfg)['completed']==result['completed']  # resume adds nothing

def test_screen_multilayer_low_lr_with_lora_qkvo(tmp_path):
    cfg=screen_v2.preset('screen_smoke')
    cfg.update(out=str(tmp_path),layers=[0,1],lr=.0005,methods=['SEPARABLE','LORA_QKVO'])
    result=screen_v2.run(cfg)
    assert result['status']=='COMPLETE' and result['completed']==result['expected']>0
    state=torch.load(tmp_path/'adapter_LORA_QKVO.pt',weights_only=True)['adapter']
    assert {k.split('_')[0] for k in state}=={'l0','l1'} and any('v_proj' in k for k in state) and any('o_proj' in k for k in state)
