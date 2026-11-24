"""Exact scoring and paired bundle-level analysis. Missing != wrong."""
import itertools
import json
import math
import random
import statistics as st
from collections import defaultdict

MODES=('READ','APPLY','NONAPPLY','MIXED')

def mean(xs):return st.mean(xs) if xs else None

def score(request,text):
    error=None
    try:
        def unique(pairs):
            obj={}
            for k,v in pairs:
                if k in obj:raise ValueError('duplicate key')
                obj[k]=v
            return obj
        obj=json.loads(text.strip(),object_pairs_hook=unique)
        if not isinstance(obj,dict):raise ValueError('not an object')
    except (ValueError,TypeError):obj=None;error='INVALID_JSON_OBJECT'
    gold=request['gold'];mode=request['mode']
    def same_shape(a,b):
        return isinstance(a,dict) and a.keys()==b.keys() and all(same_shape(a[k],v) if isinstance(v,dict) else isinstance(a[k],str) for k,v in b.items())
    out=dict(schema_error=int(error is None and not same_shape(obj,gold)),correct=int(obj==gold),parse_error=int(error is not None),parse_reason=error,
             rule_leakage=0,read_base=None,read_rule=None,mixed_read=None,mixed_apply=None,mixed_nonapply=None)
    if mode=='READ':
        out['read_base']=int(isinstance(obj,dict) and obj.get('decoded_value')==gold['decoded_value'])
        out['read_rule']=int(isinstance(obj,dict) and obj.get('rule_result')==gold['rule_result'])
    if mode=='NONAPPLY':out['rule_leakage']=int(isinstance(obj,dict) and obj.get('answer')==request['rule_answer'])
    if mode=='MIXED':
        for k in ('read','apply','nonapply'):out['mixed_'+k]=int(isinstance(obj,dict) and obj.get(k)==gold[k])
    return out

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['method'],r['seed'],r['split'])].append(r)
    panels=[]
    numeric=('input_tokens','output_tokens','latency_s','prefill_s','decode_s','peak_vram_bytes','trainable_params','peak_incremental_vram_bytes','decode_tokens_per_s')
    for (method,seed,split),rs in sorted(groups.items()):
        p=dict(method=method,seed=seed,split=split,n=len(rs),accuracy=mean([r['correct'] for r in rs]),parse_error_rate=mean([r.get('parse_error',0) for r in rs]))
        for mode in MODES:
            sub=[r for r in rs if r['mode']==mode];p[mode.lower()+'_n']=len(sub);p[mode.lower()+'_accuracy']=mean([r['correct'] for r in sub])
        p['schema_error_rate']=mean([r.get('schema_error',0) for r in rs])
        p['max_token_rate']=mean([int(r.get('termination')=='MAX_NEW_TOKENS') for r in rs])
        p['rule_leakage_rate']=mean([r.get('rule_leakage',0) for r in rs if r['mode']=='NONAPPLY'])
        for key in ('read_base','read_rule','mixed_read','mixed_apply','mixed_nonapply')+numeric:
            p[key]=mean([r[key] for r in rs if r.get(key) is not None])
        bs=defaultdict(dict)
        for r in rs:bs[(r['bundle_id'],r.get('repeat',0))][r['mode']]=r['correct']
        products=[int(all(x[m] for m in MODES[:3])) for x in bs.values() if all(m in x for m in MODES[:3])]
        p['jss']=mean(products);p['jss_n']=len(products)
        for key in ('input_tokens','output_tokens','latency_s'):
            p[key+'_total']=sum(r.get(key,0) or 0 for r in rs)
        p['peak_vram_bytes_max']=max([r['peak_vram_bytes'] for r in rs if r.get('peak_vram_bytes') is not None],default=None)
        p['all4']=mean([int(all(x[m] for m in MODES)) for x in bs.values() if all(m in x for m in MODES)])
        panels.append(p)
    return panels

def contrast(rows,a,b,bootstrap=1000,seed=123,mode=None,split=None):
    filt=[r for r in rows if (mode is None or r['mode']==mode) and (split is None or r['split']==split)]
    maps={x:{(r['seed'],r['split'],r['bundle_id'],r['mode'],r.get('repeat',0)):r for r in filt if r['method']==x} for x in (a,b)}
    common=sorted(maps[a].keys() & maps[b].keys());clusters=defaultdict(list);harm=[];rescue=[]
    for k in common:
        aa=maps[a][k]['correct'];bb=maps[b][k]['correct']
        clusters[k[1:3]].append(aa-bb);harm.append(int(aa<bb));rescue.append(int(aa>bb))
    diffs=[mean(v) for v in clusters.values()]; estimate=mean(diffs); ci=None
    if diffs and bootstrap:
        rng=random.Random(seed); samples=sorted(mean(rng.choices(diffs,k=len(diffs))) for _ in range(bootstrap))
        ci=[samples[int(.025*(bootstrap-1))],samples[int(.975*(bootstrap-1))]]
    return dict(method_a=a,method_b=b,mode=mode,split=split,paired_requests=len(common),paired_bundles=len(diffs),
                delta_accuracy=estimate,ci95=ci,negative_transfer_rate=mean(harm),rescue_rate=mean(rescue),
                ci_unit='split/bundle cluster including all seeds/repeats; conditional on tested seeds')

def jss_contrast(rows,a,b,bootstrap=1000,split=None):
    grouped=defaultdict(dict)
    for r in rows:
        if split is not None and r['split']!=split:continue
        grouped[(r['method'],r['seed'],r['split'],r['bundle_id'],r.get('repeat',0))][r['mode']]=r['correct']
    converted=[]
    for (method,seed,sp,bundle,repeat),modes in grouped.items():
        if all(x in modes for x in MODES[:3]):
            converted.append(dict(method=method,seed=seed,split=sp,bundle_id=bundle,repeat=repeat,mode='JSS',correct=int(all(modes[x] for x in MODES[:3]))))
    result=contrast(converted,a,b,bootstrap=bootstrap,split=split)
    result['delta_jss']=result.pop('delta_accuracy');return result

def research_questions(rows,algebra,expected_bundles=0):
    full=[r for r in rows if r['method']=='FULL'];models=set(r['method'] for r in rows)
    tests=[r for r in rows if r['split'].startswith('test')]
    base={'technical_validity':True,'coverage':'NOT_RUN','supported_claim':'No measurement yet.'}
    rq={f'RQ{i}':dict(base) for i in range(1,5)}
    rq['RQ1'].update(coverage='NUMERIC_CHECKS_ONLY' if algebra else 'NOT_RUN',evidence=algebra,
                     supported_claim='Exact-equivalence checks verified numerically. No expressivity theorem or native-circuit localization is established.' if algebra else 'Algebra checks not run.')
    rq['RQ2'].update(coverage='PARTIAL' if full else 'NOT_RUN',evidence={'full_accuracy':mean([r['correct'] for r in full]),'full_nonapply_leakage':mean([r.get('rule_leakage',0) for r in full if r['mode']=='NONAPPLY'])})
    if full:
        n=sum(r['mode']=='NONAPPLY' for r in full);leak=sum(r.get('rule_leakage',0) for r in full if r['mode']=='NONAPPLY')
        rq['RQ2']['observed_result']='TARGET_FAILURE_OBSERVED' if leak else 'NO_TARGET_FAILURE_OBSERVED'
        rq['RQ2']['supported_claim']=f'Observed {leak}/{n} wrong-rule NONAPPLY outputs. Zero errors means no observed target failure here, not an implementation bug.'
        rq['RQ2']['memory_content_contrasts']=[contrast(tests,'FULL',x,mode='NONAPPLY') for x in ('SCOPE_PROMPT','FACTS_ONLY','FACTS_MATCHED','FACTS_MATCHED_ALT') if x in models]
        if 'VECTOR_PAIR' in models:
            rq['RQ2']['intervention_contrasts']=[contrast(tests,'VECTOR_PAIR','FULL',mode=mode) for mode in MODES]
            rq['RQ2']['jss_contrast']=jss_contrast(tests,'VECTOR_PAIR','FULL')
    trained=('QUERY','KEY','STATIC','SEPARABLE','SCALAR_PAIR','VECTOR_PAIR','POST_GATE','LORA_V','LORA_O','LORA_QK','RESIDUAL','AUX_ATTENTION')
    present=[x for x in trained if x in models]
    if 'VECTOR_PAIR' in models:
        controls=[x for x in present if x!='VECTOR_PAIR']
        rq['RQ3'].update(coverage='PARTIAL',
                        evidence={'available_controls':controls,'missing_controls':[x for x in trained if x not in models]},
                        contrasts=[jss_contrast(tests,'VECTOR_PAIR',x) for x in controls],
                        supported_claim='A conditioning claim requires a paired gain over the trained query/separable/scalar/capacity controls, plus preserved READ/APPLY. These are empirical estimates, not a universal superiority claim.')
    if 'VECTOR_PAIR' in models:
        cs=rq['RQ3']['contrasts'];points=[x['delta_jss'] for x in cs if x['delta_jss'] is not None]
        rq['RQ3']['observed_result']=('POSITIVE_POINT_ESTIMATE_OVER_AVAILABLE_CONTROLS' if points and all(x>0 for x in points)
                                      else 'NO_ADVANTAGE_OVER_ALL_AVAILABLE_CONTROLS' if points else 'NO_COMPLETE_PAIRED_BUNDLES')
        rq['RQ3']['confirmatory_superiority']='NOT_ESTABLISHED: exploratory comparisons, capacity/cost and missing-control checks still apply'
    ood=[r for r in tests if r['split']=='test_ood'];seeds=sorted(set(r['seed'] for r in ood if r['method']=='VECTOR_PAIR'))
    if ood:
        rq['RQ4'].update(coverage='PARTIAL' if len(seeds)<3 else 'MULTISEED_SYNTHETIC_ONLY',
                        evidence={'seeds':seeds,'ood_requests':len(ood),'external_benchmarks':'NOT_RUN','second_model':'NOT_RUN'},
                        contrasts=[jss_contrast(tests,'VECTOR_PAIR',x,split='test_ood') for x in ('FULL','QUERY','SEPARABLE') if x in models and 'VECTOR_PAIR' in models],
                        supported_claim='Held-out cue families only. A pilot cannot establish cross-model or natural-task generalization.')
    for value in rq.values():
        value['expected_test_bundles_per_seed']=expected_bundles
        if value['coverage']=='NOT_RUN':value['technical_validity']=None
    return rq
