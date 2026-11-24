"""Write the Colab job spec that reproduces every experiment in the paper, plus the matching analysis layout.

    python tools/make_job_specs.py            -> deploy/reproduce_all/job.json, paper/layout_reproduce.json
    python tools/make_job_specs.py --round5   -> deploy/colab_round5/job.json (round 5 only; --round6 likewise)

Run the spec with tools/colab_job.py (one GPU, stages in parallel); it writes results_reproduce_all.zip. Put that zip at
runs/reproduce_all/results_reproduce_all.zip and run: python paper/analysis.py --layout paper/layout_reproduce.json
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS={'Qwen3-4B':('qwen3_4b','Qwen/Qwen3-4B-Instruct-2507'),'Qwen3-1.7B':('qwen3_1p7b','Qwen/Qwen3-1.7B'),
        'Qwen2.5-3B':('qwen25_3b','Qwen/Qwen2.5-3B-Instruct'),'Phi-3.5-mini':('phi35_mini','microsoft/Phi-3.5-mini-instruct')}
PHI='microsoft/Phi-3.5-mini-instruct'
SEEDS=(17,29,43,7,11)
ANY={'allow_any_arch':True}
E3_COMMON={'methods':['LORA_QKVO','SEPARABLE','VECTOR_PAIR'],'layers':[6,12,18,24]}
BOTH_ORDERS=[[f,d,o] for f in ('neutral','default') for d in ('none','q') for o in ('book_first','rule_first')]
MECH=['M0','M1','M2','M3','M4','M5']

def stage(label,module,preset,**overrides):
    return dict(label=label,module='memlearn_attention.'+module,preset=preset,overrides=overrides)

NEW={'OLMo-2-7B':('olmo2_7b','allenai/OLMo-2-1124-7B-Instruct'),'Granite-3.3-8B':('granite33_8b','ibm-granite/granite-3.3-8b-instruct'),
     'Qwen3-14B':('qwen3_14b','Qwen/Qwen3-14B')}
Q8=('qwen3_8b','Qwen/Qwen3-8B')

def round4_stages():
    """Round 4 (docs/ROUND4_PLAN.md): circuit, prose mechanism, new families, extra prose templates."""
    s=[]
    for tag,model in (MODELS['Qwen3-4B'],Q8,MODELS['Phi-3.5-mini']):s.append(stage(f'e4c_{tag}','mech_e4','mech_e4',model=model,**ANY))
    for tag,model in (MODELS['Qwen3-4B'],Q8):
        for style,ds,order in (('nl',41,'book_first'),('nl',41,'rule_first'),('codes',17,'rule_first')):
            s.append(stage(f'e4_{style}_{"bf" if order=="book_first" else "rf"}_{tag}','mech_e4','mech_e4',model=model,style=style,order=order,data_seed=ds,**ANY))
    for name,(tag,model) in NEW.items():
        s.append(stage(f'probe_{tag}','probe_v2','probe_v2',model=model,**ANY))
        s.append(stage(f'mech_{tag}','mech_v2','mech_v2',model=model,variants=MECH,**ANY))
        s.append(stage(f'nl_probe_{tag}','probe_v2','probe_v2',model=model,style='nl',data_seed=41,**ANY))
    for style in ('nl_gate','nl_shelf'):
        for tag,model in (MODELS['Qwen3-4B'],Q8,MODELS['Phi-3.5-mini'],NEW['OLMo-2-7B'],NEW['Granite-3.3-8B']):
            s.append(stage(f'{style}_probe_{tag}','probe_v2','probe_v2',model=model,style=style,data_seed=53,**ANY))
            s.append(stage(f'{style}_mech_{tag}','mech_v2','mech_v2',model=model,style=style,data_seed=53,variants=MECH,**ANY))
    return s

R5={'Gemma-2-9B':('gemma2_9b','google/gemma-2-9b-it'),'Gemma-2-2B':('gemma2_2b','google/gemma-2-2b-it'),
    'Mistral-7B':('mistral_7b','mistralai/Mistral-7B-Instruct-v0.3')}
BF16={'dtype':'bfloat16'}  # Gemma-2 overflows in float16
# Round 5 path runs: (label, model, style, path layer L*). L* = 16 for Qwen3 (the shared restoration window, ROUND4 G1), 12 for Phi
# (round-3 E4), 14 for Mistral (0.44 of depth, as Qwen3). Gemma-2 has post-norms, so only the receiver and route parts run.
PATH_RUNS=(('path_codes_qwen3_4b',MODELS['Qwen3-4B'][1],'codes',16),('path_codes_qwen3_8b',Q8[1],'codes',16),
           ('path_codes_phi35_mini',PHI,'codes',12),('path_codes_mistral_7b',R5['Mistral-7B'][1],'codes',14),
           ('path_codes_gemma2_9b',R5['Gemma-2-9B'][1],'codes',16),
           ('path_nl_qwen3_4b',MODELS['Qwen3-4B'][1],'nl',16),('path_nl_qwen3_8b',Q8[1],'nl',16),
           ('path_shelf_qwen3_4b',MODELS['Qwen3-4B'][1],'nl_shelf',16),('path_shelf_qwen3_8b',Q8[1],'nl_shelf',16))

def vram_gb(model,label):
    """Rough peak VRAM of one stage (weights + attention maps), for colab_job's vram_budget_gb scheduling."""
    size=30 if '9b' in model else 24 if ('8B' in model or '7B' in model) else 10 if '2b' in model else 12
    return size+(4 if label.startswith(('path_','e4c_')) and size>12 else 0)

def round5_stages():
    """Round 5 (docs/ROUND5_PLAN.md): sender/receiver path decomposition, the prose route, Gemma-2 and Mistral."""
    s=[]
    for label,model,style,L in PATH_RUNS:
        s.append(dict(stage(label,'mech_path','mech_path',model=model,style=style,path_layer=L,**ANY,**(BF16 if 'gemma' in model else {})),budget_seconds=3600))
    for name,(tag,model) in R5.items():
        extra=dict(ANY,**(BF16 if 'gemma' in model else {}))
        s.append(stage(f'probe_{tag}','probe_v2','probe_v2',model=model,**extra))
        s.append(stage(f'mech_{tag}','mech_v2','mech_v2',model=model,variants=MECH,**extra))
        s.append(stage(f'nl_probe_{tag}','probe_v2','probe_v2',model=model,style='nl',data_seed=41,**extra))
        if tag!='gemma2_2b':s.append(stage(f'e4c_{tag}','mech_e4','mech_e4',model=model,**extra))
    big=[];small=[]
    for st in s:st['vram_gb']=vram_gb(st['overrides']['model'],st['label']);(big if st['vram_gb']>12 else small).append(st)
    # Alternate large and small stages so the VRAM budget keeps the GPU busy.
    return [x for pair in zip(big,small+[None]*len(big)) for x in pair if x]

# Llama-3 through the ungated unsloth mirrors (same weights; Llama 3.1/3.2 Community License applies).
R6={'Llama-3.1-8B':('llama31_8b','unsloth/Llama-3.1-8B-Instruct'),'Llama-3.2-3B':('llama32_3b','unsloth/Llama-3.2-3B-Instruct')}
# Round 6 per-query-head key runs: fresh bundles (data 61, heads ranked on 73), receiver + heads parts only.
HEADS6=(('heads_codes_qwen3_4b',MODELS['Qwen3-4B'][1],'codes',16),('heads_codes_qwen3_8b',Q8[1],'codes',16),('heads_codes_phi35_mini',PHI,'codes',12),
        ('heads_codes_mistral_7b',R5['Mistral-7B'][1],'codes',14),('heads_codes_gemma2_9b',R5['Gemma-2-9B'][1],'codes',16),
        ('heads_codes_llama31_8b',R6['Llama-3.1-8B'][1],'codes',14),('heads_nl_qwen3_4b',MODELS['Qwen3-4B'][1],'nl',16))

def round6_stages():
    """Round 6 (docs/ROUND6_PLAN.md): keys per query head, and Llama-3."""
    s=[]
    for label,model,style,L in HEADS6:
        s.append(dict(stage(label,'mech_path','mech_path',model=model,style=style,path_layer=L,data_seed=61,head_seed=73,parts=['receiver','heads'],
                            **ANY,**(BF16 if 'gemma' in model else {})),budget_seconds=3600))
    tag,model=R6['Llama-3.1-8B']
    s.append(dict(stage(f'path_codes_{tag}','mech_path','mech_path',model=model,style='codes',path_layer=14,parts=['sender','receiver','route','heads'],**ANY),budget_seconds=3600))
    for name,(tag,model) in R6.items():
        s.append(stage(f'probe_{tag}','probe_v2','probe_v2',model=model,**ANY))
        s.append(stage(f'mech_{tag}','mech_v2','mech_v2',model=model,variants=MECH,**ANY))
        s.append(stage(f'nl_probe_{tag}','probe_v2','probe_v2',model=model,style='nl',data_seed=41,**ANY))
        if tag=='llama31_8b':s.append(stage(f'e4c_{tag}','mech_e4','mech_e4',model=model,**ANY))
    big=[];small=[]
    for st in s:st['vram_gb']=vram_gb(st['overrides']['model'],st['label']);(big if st['vram_gb']>12 else small).append(st)
    return [x for pair in zip(big,small+[None]*len(big)) for x in pair if x]

def round6_job():
    return dict(name='round6',parallel=5,vram_budget_gb=84,budget_seconds=2400,models=sorted({st['overrides']['model'] for st in round6_stages()}),
                stages=round6_stages(),plan='docs/ROUND6_PLAN.md')

def reproduce_all():
    s=[]
    for name,(tag,model) in MODELS.items():  # E1/B: frozen behaviour and knockout
        s.append(stage(f'probe_{tag}','probe_v2','probe_v2',model=model,**ANY))
        s.append(stage(f'mech_{tag}','mech_v2','mech_v2',model=model,variants=MECH+(['M6'] if tag=='qwen3_4b' else []),**ANY))
    s.append(stage('e2_qwen3_4b','mech_v2','mech_e2'))  # E2: narrow knockouts
    s.append(stage('full','screen_v2','screen_v2',methods=['FULL']))  # E3/E7: adaptation, 5 seeds
    for seed in SEEDS:
        for mode,lr in (('bf',5e-4),('both',5e-4),('bfhi',2e-3),('lr1e3',1e-3)):
            ov=dict(E3_COMMON,seed=seed,lr=lr)
            if mode=='both':ov['train_conditions']=BOTH_ORDERS
            s.append(stage(f'{mode}_s{seed}','screen_v2','screen_v2',**ov))
        s.append(stage(f'q1l_s{seed}','screen_v2','screen_v2',seed=seed,layers='middle',lr=2e-3,chunk=16,methods=['QUERY','SEPARABLE','VECTOR_PAIR','LORA_V']))
        s.append(stage(f'phi_s{seed}','screen_v2','screen_v2',model=PHI,data_seed=23,layers=[5,11,16,21],lr=5e-4,chunk=64,seed=seed,
                       methods=['VECTOR_PAIR','QUERY','SEPARABLE','LORA_QKVO'],**ANY))
    s.append(stage('phi_full','screen_v2','screen_v2',model=PHI,data_seed=23,layers=[5,11,16,21],methods=['FULL'],**ANY))
    s.append(stage('e4_qwen3_4b','mech_e4','mech_e4'));s.append(stage('e4_phi35_mini','mech_e4','mech_e4',model=PHI))  # E4
    for name,(tag,model) in MODELS.items():  # E5: natural language + free generation
        s.append(stage(f'nl_probe_{tag}','probe_v2','probe_v2',model=model,style='nl',data_seed=41,**ANY))
        s.append(stage(f'nl_mech_{tag}','mech_v2','mech_v2',model=model,style='nl',data_seed=41,variants=MECH,**ANY))
        s.append(stage(f'gen_{tag}','probe_v2','probe_v2',model=model,generate_every=4,**ANY))
    s.append(stage('q8b_probe','probe_v2','probe_v2',model='Qwen/Qwen3-8B',**ANY))  # E6
    s.append(stage('q8b_mech','mech_v2','mech_v2',model='Qwen/Qwen3-8B',variants=MECH,**ANY))
    s.append(stage('filler_phi35_mini','probe_v2','probe_v2',model=PHI,data_seed=31,  # E8
                   conditions=[[f,d,'book_first'] for f in ('neutral','default','filler') for d in ('none','q')],**ANY))
    s+=round4_stages()+round5_stages()+round6_stages()
    return dict(name='reproduce_all',parallel=8,budget_seconds=2400,stages=s,
                models=sorted({m for _,m in MODELS.values()}|{'Qwen/Qwen3-8B'}|{m for _,m in NEW.values()}|{m for _,m in R5.values()}|{m for _,m in R6.values()}),
                note=f'{len(s)} stages; about 4.5 h on one 96GB GPU with parallel=8 (lower parallel on smaller GPUs: ~10GB VRAM per stage, '
                     '~20GB for 8B-9B). Gemma-2 is gated: accept its license on Hugging Face and export HF_TOKEN before running.')

def round5_job():
    return dict(name='round5',parallel=5,vram_budget_gb=84,budget_seconds=2400,models=sorted({m for _,m,_,_ in PATH_RUNS}|{m for _,m in R5.values()}),
                stages=round5_stages(),plan='docs/ROUND5_PLAN.md')

def layout_reproduce(zip_rel='reproduce_all/results_reproduce_all.zip'):
    z=lambda label,kind='probe':{'zip':zip_rel,'kind':kind,'label':label}
    t={n:tag for n,(tag,_) in MODELS.items()}
    return dict(frozen={n:z(f'probe_{x}') for n,x in t.items()},mech={n:z(f'mech_{x}','mech') for n,x in t.items()},e2=z('e2_qwen3_4b','mech'),
                adaptation=[{'zip':zip_rel,'kind':'probe','prefixes':['full','bf_s','both_s','bfhi_s','lr1e3_s','q1l_s','phi_s']}],
                e4={'Qwen3-4B':z('e4_qwen3_4b','e4'),'Phi-3.5-mini':z('e4_phi35_mini','e4')},
                generation={n:z(f'gen_{x}') for n,x in t.items()},nl_probe={n:z(f'nl_probe_{x}') for n,x in t.items()},
                nl_mech={n:z(f'nl_mech_{x}','mech') for n,x in t.items()},scale_probe={'Qwen3-8B':z('q8b_probe')},
                scale_mech={'Qwen3-8B':z('q8b_mech','mech')},framing=z('filler_phi35_mini'),
                e4_circuit={'Qwen3-4B':z('e4c_qwen3_4b','e4'),'Qwen3-8B':z('e4c_qwen3_8b','e4'),'Phi-3.5-mini':z('e4c_phi35_mini','e4')},
                e4_format={f'{n}|{f}|{o}':z(lab,'e4') for n,tag in (('Qwen3-4B','qwen3_4b'),('Qwen3-8B','qwen3_8b'))
                           for f,o,lab in (('codes','book_first',f'e4c_{tag}'),('codes','rule_first',f'e4_codes_rf_{tag}'),
                                           ('nl','book_first',f'e4_nl_bf_{tag}'),('nl','rule_first',f'e4_nl_rf_{tag}'))},
                families_probe={n:z(f'probe_{t}') for n,(t,_) in NEW.items()},families_mech={n:z(f'mech_{t}','mech') for n,(t,_) in NEW.items()},
                families_nl={n:z(f'nl_probe_{t}') for n,(t,_) in NEW.items()},
                **{f'{st}_{k}':{n:z(f'{st}_{k}_{t}',kind) for n,t in (('Qwen3-4B','qwen3_4b'),('Qwen3-8B','qwen3_8b'),('Phi-3.5-mini','phi35_mini'),
                                                                     ('OLMo-2-7B','olmo2_7b'),('Granite-3.3-8B','granite33_8b'))}
                   for st in ('nl_gate','nl_shelf') for k,kind in (('probe','probe'),('mech','mech'))},
                path={lab[5:]:z(lab,'path') for lab,_,_,_ in PATH_RUNS},path_mass={lab[5:]:z(lab,'pathmass') for lab,_,_,_ in PATH_RUNS},
                families5_probe={n:z(f'probe_{t}') for n,(t,_) in R5.items()},families5_mech={n:z(f'mech_{t}','mech') for n,(t,_) in R5.items()},
                families5_nl={n:z(f'nl_probe_{t}') for n,(t,_) in R5.items()},
                e4_circuit5={n:z(f'e4c_{t}','e4') for n,(t,_) in R5.items() if t!='gemma2_2b'},
                heads={lab[6:]:z(lab,'path') for lab,_,_,_ in HEADS6},path_llama={'codes_llama31_8b':z('path_codes_llama31_8b','path')},
                path_llama_mass={'codes_llama31_8b':z('path_codes_llama31_8b','pathmass')},
                families6_probe={n:z(f'probe_{t}') for n,(t,_) in R6.items()},families6_mech={n:z(f'mech_{t}','mech') for n,(t,_) in R6.items()},
                families6_nl={n:z(f'nl_probe_{t}') for n,(t,_) in R6.items()},e4_circuit6={'Llama-3.1-8B':z('e4c_llama31_8b','e4')})

if __name__=='__main__':
    import sys
    if '--round6' in sys.argv:
        out=ROOT/'deploy'/'colab_round6';out.mkdir(parents=True,exist_ok=True);(out/'job.json').write_text(json.dumps(round6_job(),indent=1))
        print('stages',len(round6_job()['stages']),'->',out/'job.json');raise SystemExit
    if '--round5' in sys.argv:
        out=ROOT/'deploy'/'colab_round5';out.mkdir(parents=True,exist_ok=True);(out/'job.json').write_text(json.dumps(round5_job(),indent=1))
        print('stages',len(round5_job()['stages']),'->',out/'job.json');raise SystemExit
    spec=reproduce_all();out=ROOT/'deploy'/'reproduce_all';out.mkdir(parents=True,exist_ok=True)
    (out/'job.json').write_text(json.dumps(spec,indent=1));(ROOT/'paper'/'layout_reproduce.json').write_text(json.dumps(layout_reproduce(),indent=1))
    print('stages',len(spec['stages']),'->',out/'job.json')
