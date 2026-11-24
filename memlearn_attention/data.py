"""Deterministic nonce tasks. Oracle and renderer are separate functions."""
import hashlib
import json
import random

TRAIN_FAMILIES=('direct','authorize','report','workflow')
DEV_FAMILIES=('permission','quotation')
TEST_FAMILIES=('jurisdiction','exception')
MODES=('READ','APPLY','NONAPPLY','MIXED')

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def make_bundle(seed,split,index,family):
    rng=random.Random(digest([seed,split,index,family]))
    # Unique whole codebooks are seed/split separated; individual symbols may recur.
    keys=rng.sample([f'x{i:02d}' for i in range(100)],4)
    values=rng.sample([f'z{i:02d}' for i in range(100)],4)
    shift=rng.randrange(1,4)
    book=dict(zip(keys,values));rule={v:values[(i+shift)%4] for i,v in enumerate(values)}
    selected=rng.choice(keys)
    order=list(('read','apply','nonapply'));rng.shuffle(order)
    return dict(id=digest([seed,split,index,family])[:20],seed=seed,split=split,
                family=family,codebook=book,rule=rule,key=selected,mixed_order=order,
                source_order=rng.choice(['book_first','rule_first']))

def dataset(n_train=24,n_dev=4,n_test=6,n_ood=6,seed=17,n_dev_ood=None):
    sizes={'train':n_train,'dev_id':n_dev,'dev_ood':n_dev if n_dev_ood is None else n_dev_ood,'test_id':n_test,'test_ood':n_ood}
    return {split:[make_bundle(seed,split,i,(TRAIN_FAMILIES if split in ('train','dev_id','test_id') else DEV_FAMILIES if split=='dev_ood' else TEST_FAMILIES)[i%(4 if split in ('train','dev_id','test_id') else 2)]) for i in range(n)] for split,n in sizes.items()}

def oracle(b,mode):
    base=b['codebook'][b['key']]; transformed=b['rule'][base]
    if mode=='READ':return {'decoded_value':base,'rule_result':transformed}
    if mode=='APPLY':return {'answer':transformed}
    if mode=='NONAPPLY':return {'answer':base}
    if mode=='MIXED':return {'read':oracle(b,'READ'),'apply':oracle(b,'APPLY'),'nonapply':oracle(b,'NONAPPLY')}
    raise ValueError(mode)

def requests_for(b):
    return [dict(id=b['id']+'_'+mode,bundle_id=b['id'],seed=b['seed'],split=b['split'],family=b['family'],
                 mode=mode,bundle=b,gold=oracle(b,mode),rule_answer=oracle(b,'APPLY')['answer']) for mode in MODES]

_CUES={
'direct':('Apply P to the decoded value.','Return the decoded value without applying P.'),
'authorize':('This request authorizes procedure P. Return its resulting value.','This request does not authorize P. Return the decoded value.'),
'report':('The report asks for the processed value under P.','The report asks for the raw decoded value.'),
'workflow':('Continue through the decoding stage and then the P stage.','Stop after the decoding stage; return that value.'),
'permission':('Permission is granted for P; compute the processed value.','Permission for P is withheld; provide only the decoded value.'),
'quotation':('Use P operationally on the decoded value.','P is background reference for this request; answer with the decoded value.'),
'jurisdiction':('P governs this request. Give the final processed value.','This request is outside the jurisdiction of P. Give the decoded value.'),
'exception':('The default processing procedure P is in force.','An exemption from P is in force for this request. Give the decoded value.')}

# v1.1: v1 named the READ field "base"; pilot30 v1 showed Qwen3-4B filling it with the
# INPUT_KEY itself (9/9 FULL READ failures), a contract ambiguity rather than a scope error.
CONTRACT=('You solve exact codebook queries. Decode an input key using CODEBOOK: the decoded value is the CODEBOOK value '
          'mapped from INPUT_KEY, never the input key itself. Procedure P replaces the decoded value using RULE_MAP. '
          'READ asks for the decoded value and the hypothetical result of P. Other requests specify whether to apply P. '
          'Return only one JSON object, no explanation or markdown. READ schema: {"decoded_value":"...","rule_result":"..."}. '
          'A single value schema: {"answer":"..."}. MIXED schema: {"read":{"decoded_value":"...","rule_result":"..."},"apply":{"answer":"..."},"nonapply":{"answer":"..."}}; '
          'in MIXED, "apply" and "nonapply" are objects with an "answer" field, not bare strings.')

def render(r,control='FULL'):
    b=r['bundle'];mode=r['mode']; family=b['family']
    book='CODEBOOK: '+canonical(b['codebook'])
    rule='RULE_MAP for P: '+canonical(b['rule'])
    if control=='FACTS_ONLY':
        # Same task, missing rule; NOT used as a fair trained baseline on APPLY/READ.
        rule='RULE_MAP for P: [not provided]'
    parts=[book,rule] if b['source_order']=='book_first' else [rule,book]
    if control=='SCOPE_PROMPT':
        parts.insert(0,'Use each requested operation only within its own scope. Knowing P does not mean every output applies P. Preserve the information needed to answer READ.')
    text='\n'.join(parts)+'\nINPUT_KEY: '+b['key']+'\n'
    yes,no=_CUES[family]
    if mode=='READ':text+='REQUEST READ: Retrieve the decoded value and the hypothetical rule_result of P.'
    elif mode=='APPLY':text+='REQUEST: '+yes
    elif mode=='NONAPPLY':text+='REQUEST: '+no
    else:
        sections={'read':'READ: retrieve the decoded value and hypothetical rule_result.','apply':yes,'nonapply':no}
        text+='REQUEST MIXED:\n'+'\n'.join(name+': '+sections[name] for name in b['mixed_order'])
    return text

def messages(r,control='FULL'):
    # No split, uid, gold, difficulty or applicability labels enter the adapter.
    return [{'role':'system','content':CONTRACT},{'role':'user','content':render(r,control)}]
