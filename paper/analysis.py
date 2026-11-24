"""Regenerate every paper table and figure from raw per-item receipts.

    python paper/analysis.py [--layout paper/layout_original.json] [--runs runs]

A layout file maps each dataset to its receipts, so the same script works on our original runs and on a fresh
reproduction (paper/layout_reproduce.json, produced by deploy/reproduce_all/job.json). A source is either
{"dir": "<receipts dir>", "stage": "<label>"} or {"zip": "<results zip>", "kind": "probe|mech|e4",
"label": "<one stage>" | "prefixes": [...]}. Missing sources are skipped with a note, never invented.
All numbers are recomputed here; CIs are paired cluster bootstraps over bundles (and seeds where present).
"""
import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent;TABLES=OUT/'tables';FIGS=OUT/'figures'
MODELS=['Qwen3-4B','Qwen3-1.7B','Qwen2.5-3B','Phi-3.5-mini']
# Reference palette (dataviz skill, light mode; first 4 slots validated with scripts/validate_palette.js). Fixed slot order.
SLOTS=['#2a78d6','#eb6834','#1baf7a','#eda100']
INK='#0b0b0b';INK2='#52514e';GRID='#e6e5e1';SURFACE='#fcfcfb'
RUNS=None;NOTES=[]

# ---------------------------------------------------------------- loading
def load(src,kind_default='probe'):
    """Rows from one layout source; each row gets r['stage']."""
    if src is None:return []
    if isinstance(src,list):return [r for s in src for r in load(s,kind_default)]
    if 'dir' in src:
        d=RUNS/src['dir']
        if not d.exists():NOTES.append(f"missing {src['dir']}");return []
        rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted(d.glob('*.json'))]
        for r in rows:r['stage']=src.get('stage',d.parent.name)
        return rows
    if 'dir_glob' in src:
        return [r for d in sorted(RUNS.glob(src['dir_glob'])) for r in load({'dir':str(d.relative_to(RUNS)),'stage':d.parent.name})]
    path=RUNS/src['zip']
    if not path.exists():NOTES.append(f"missing {src['zip']}");return []
    kind=src.get('kind',kind_default);out=[]
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            parts=n.split('/')
            if len(parts)==4 and parts[2]==kind and n.endswith('.json'):
                label=parts[1]
                if 'label' in src and label!=src['label']:continue
                if 'prefixes' in src and not any(label.startswith(p) for p in src['prefixes']):continue
                r=json.loads(z.read(n))['row'];r['stage']=label;out.append(r)
    return out

def zip_json(src,name):
    """A JSON side file (e.g. head_scores.json) from a zip stage or a dir's parent."""
    if src is None:return None
    if 'dir' in src:
        p=RUNS/src['dir']/'..'/name
        return json.loads(p.read_text()) if p.exists() else None
    path=RUNS/src['zip']
    if not path.exists():return None
    with zipfile.ZipFile(path) as z:
        n=f'runs/{src["label"]}/{name}'
        return json.loads(z.read(n)) if n in z.namelist() else None

# ---------------------------------------------------------------- helpers
def acc(rs,key='correct'):return sum(r[key] for r in rs)/len(rs) if rs else float('nan')

def pick(rs,**kw):
    def ok(r):
        for k,v in kw.items():
            if k=='order':
                if r['condition'][2]!=v:return False
            elif k=='framing':
                if r['condition'][0]!=v:return False
            elif k=='distractor':
                if r['condition'][1]!=v:return False
            elif r.get(k)!=v:return False
        return True
    return [r for r in rs if ok(r)]

def boot(per,n=4000,seed=0):
    rng=random.Random(seed);bs=sorted(sum(per[rng.randrange(len(per))] for _ in per)/len(per) for _ in range(n))
    return sum(per)/len(per),bs[int(.025*n)],bs[int(.975*n)]

def paired(rows_a,rows_b,key='correct',match=('bundle_id','split','mode'),cluster=('bundle_id','split')):
    """Mean a-b over uniquely matched items; reject silently overwritten controls.

    Condition fields can be named individually to keep every fixed factor in a
    contrast's matching key. Modes differ for READ/APPLY; orders differ for bf/rf.
    """
    def field(r,k):
        if k in ('framing','distractor','order'):
            return r['condition'][('framing','distractor','order').index(k)]
        return r.get(k)
    def index(rows):
        out={}
        for r in rows:
            ident=tuple(field(r,k) for k in match)
            if ident in out:raise ValueError(f'Duplicate paired key {ident!r}; include all fixed conditions in match')
            out[ident]=r
        return out
    left=index(rows_a);b=index(rows_b);units=defaultdict(list)
    for ident,r in left.items():
        o=b.get(ident)
        if o is not None:units[tuple(r.get(k) for k in cluster)].append(r[key]-o[key])
    per=[sum(v)/len(v) for v in units.values()]
    return boot(per) if per else (float('nan'),)*3

def ci(t):return f'{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]'

def write_table(name,header,rows,note=''):
    with (TABLES/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows([header]+rows)
    fmt=lambda x:(f'{x:.2f}' if isinstance(x,float) else str(x)).replace('|','\\|')
    md=['| '+' | '.join(fmt(h) for h in header)+' |','|'+'|'.join(':---' if i==0 else '---:' for i in range(len(header)))+'|']
    md+=['| '+' | '.join(fmt(x) for x in r)+' |' for r in rows]
    (TABLES/f'{name}.md').write_text('\n'.join(md)+('\n\n'+note if note else '')+'\n',encoding='utf-8')

def leak_str(rs):return f"{sum(r['leakage'] for r in rs)}/{len(rs)}"

# ---------------------------------------------------------------- tables
def t1_frozen(frozen,name='t1_frozen',note=''):
    out=[]
    for m,rs in frozen.items():
        if not rs:continue
        na=pick(rs,mode='NONAPPLY');ap=pick(rs,mode='APPLY_P')
        out.append([m,acc(pick(rs,mode='READ_DEC',order='book_first')),acc(pick(rs,mode='READ_RULE',order='book_first')),acc(pick(rs,mode='READ_RULE',order='rule_first')),
                    acc(pick(rs,mode='APPLY_P',order='book_first')),acc(pick(rs,mode='APPLY_P',order='rule_first')),
                    sum(r['choice']=='base' for r in ap)/len(ap),leak_str(pick(na,order='book_first')),leak_str(pick(na,order='rule_first'))])
    write_table(name,['Model','READ_DEC bf','READ_RULE bf','READ_RULE rf','APPLY bf','APPLY rf','APPLY stop-at-base','Leak bf','Leak rf'],out,
                note or 'Frozen models, 4-candidate log-prob probe; bf = codebook first, rf = rule first.')

def t2_knockout(mech,name='t2_knockout_read_rule',note=''):
    out=[]
    for m,rs in mech.items():
        if not rs:continue
        line=[m]+[acc(pick(rs,variant=v,mode='READ_RULE')) for v in ('M0','M2','M1','M3','M5','M4')]
        line.append(ci(paired(pick(rs,variant='M1',mode='READ_RULE'),pick(rs,variant='M2',mode='READ_RULE'))))
        out.append(line)
    write_table(name,['Model','bf none','bf sham','bf rule-x->book','rf none','rf sham','rf book-x->rule','Block-sham (bf) [CI95]'],out,
                note or 'READ_RULE accuracy (one-hop rule lookup, decoded value given) under attention blocks in all layers.')

def t3_e2(rs):
    if not rs:return
    label={'M2':'sham, all layers','M1':'codebook, all layers','N3':'codebook, layers 0-11','N4':'codebook, layers 12-23',
           'N5':'codebook, layers 24-35','N6':'codebook keys x.. only','N7':'codebook values z.. only'}
    out=[[label[v],ci(paired(pick(rs,variant=v,mode='APPLY_P'),pick(rs,variant='M0',mode='APPLY_P'))),
          ci(paired(pick(rs,variant=v,mode='READ_RULE'),pick(rs,variant='M0',mode='READ_RULE'))),acc(pick(rs,variant=v,mode='NONAPPLY'))]
         for v in ('M2','M1','N3','N4','N5','N6','N7')]
    write_table('t3_e2_narrow',['Rule tokens blocked from','dAPPLY vs none','dREAD_RULE vs none','NONAPPLY'],out,
                f"Qwen3-4B, codebook first. Reference: APPLY {acc(pick(rs,variant='M0',mode='APPLY_P')):.2f}, READ_RULE {acc(pick(rs,variant='M0',mode='READ_RULE')):.2f}.")

SETTINGS=[('Qwen3-4B, 4 layers, lr 5e-4, train bf','bf_s'),('Qwen3-4B, 4 layers, lr 1e-3, train bf','lr1e3_s'),('Qwen3-4B, 4 layers, lr 2e-3, train bf','bfhi_s'),
          ('Qwen3-4B, 4 layers, lr 5e-4, train bf+rf','both_s'),('Qwen3-4B, 1 layer, lr 2e-3, train bf','q1l_s'),('Phi-3.5-mini, 4 layers, lr 5e-4, train bf','phi_s')]
LABEL={'QUERY':'Query','SEPARABLE':'Separable','VECTOR_PAIR':'Vector pair','LORA_QKVO':'LoRA-qkvo','LORA_V':'LoRA-v'}

def t4_adaptation(rows):
    out=[];leak_ci=[]
    for title,prefix in SETTINGS:
        sub_all=[r for r in rows if r['stage'].startswith(prefix)]
        for m in ('QUERY','SEPARABLE','VECTOR_PAIR','LORA_QKVO','LORA_V'):
            sub=[r for r in sub_all if r['method']==m]
            if not sub:continue
            seeds=sorted({r['stage'] for r in sub});na=pick(sub,mode='NONAPPLY')
            rf=[acc(pick([r for r in sub if r['stage']==s],mode='APPLY_P',order='rule_first')) for s in seeds]
            out.append([title,LABEL[m],len(seeds),acc(pick(sub,mode='APPLY_P',order='book_first')),acc(pick(sub,mode='APPLY_P',order='rule_first')),
                        f'{min(rf):.2f}-{max(rf):.2f}',acc(pick(sub,mode='READ_RULE')),leak_str(na),acc(na,'leakage')])
        # Pairwise leakage contrasts within the setting: clusters = seed x bundle.
        ms=[m for m in ('QUERY','SEPARABLE','VECTOR_PAIR','LORA_QKVO','LORA_V') if any(r['method']==m for r in sub_all)]
        for i,a in enumerate(ms):
            for b in ms[i+1:]:
                ra=pick([r for r in sub_all if r['method']==a],mode='NONAPPLY');rb=pick([r for r in sub_all if r['method']==b],mode='NONAPPLY')
                t=paired(ra,rb,key='leakage',match=('stage','id'),cluster=('stage','bundle_id','split'))
                leak_ci.append([title,f'{LABEL[a]} - {LABEL[b]}',f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'])
    write_table('t4_adaptation',['Setting','Method','Seeds','APPLY bf','APPLY rf','APPLY rf (seed range)','READ_RULE','Leak (NONAPPLY)','Leak rate'],out,
                'Adapters on frozen backbones, parameter-matched, 256 steps; evaluation neutral framing x {none,q} x {bf,rf}. Training never sees rf unless stated.')
    write_table('t4b_leak_contrasts',['Setting','Contrast','Leak-rate difference [CI95]'],leak_ci,'Paired by item within seed; bootstrap over seed x bundle clusters.')

def t5_e4(e4):
    out=[];curves={}
    for m,(rs,hs) in e4.items():
        if not rs:continue
        clean={mode:acc(pick(rs,variant='clean',mode=mode)) for mode in ('APPLY_P','READ_RULE')}
        blocked={mode:acc(pick(rs,variant='blocked',mode=mode)) for mode in ('APPLY_P','READ_RULE')}
        layers=sorted({r['restore_layer'] for r in rs if r['variant'].startswith('restore_L')})
        curves[m]={mode:[acc(pick(rs,variant=f'restore_L{l}',mode=mode)) for l in layers] for mode in ('APPLY_P','READ_RULE')};curves[m]['layers']=layers
        curves[m]['clean']=clean;curves[m]['blocked']=blocked
        first={mode:next((l for l,a in zip(layers,curves[m][mode]) if a>=clean[mode]-.10),None) for mode in ('APPLY_P','READ_RULE')}
        line=[m,clean['APPLY_P'],blocked['APPLY_P'],str(first['APPLY_P']),clean['READ_RULE'],blocked['READ_RULE'],str(first['READ_RULE'])]
        knockout=[]
        for k in sorted({int(v[3:]) for v in {r['variant'] for r in rs} if v.startswith('top')}):
            for mode in ('APPLY_P','READ_RULE'):
                top=pick(rs,variant=f'top{k}',mode=mode);rnd=[r for r in rs if r['variant'].startswith(f'rand{k}_') and r['mode']==mode]
                # Top-k minus mean random-k, matched per item (random draws averaged first).
                avg=defaultdict(list)
                for r in rnd:avg[(r['bundle_id'],r['split'],r['mode'])].append(r['correct'])
                rr=[dict(bundle_id=b,split=s,mode=md,correct=sum(v)/len(v)) for (b,s,md),v in avg.items()]
                knockout.append(f'{mode} top{k} {acc(top):.2f} vs rand {acc(rr):.2f}: {ci(paired(top,rr))}')
        line.append('; '.join(knockout))
        out.append(line)
    write_table('t5_e4',['Model','APPLY clean','APPLY blocked','APPLY recovered by L','READ clean','READ blocked','READ recovered by L','Head knockout (APPLY, READ) per k'],out,
                'Restoration: block rule->codebook in all layers, restore clean rule-token states after layer L ("recovered by L" = first L within 0.10 of clean). Heads: duplicate-token score top-k (first 2/3 of layers) vs mean of random-k draws, rule -> codebook values.')
    return curves

def t6_generation(gen):
    out=[]
    for m,rs in gen.items():
        g=[r for r in rs if 'gen_choice' in r]
        if not g:continue
        valid=[r for r in g if r['gen_choice']!='invalid']
        gacc=lambda mode,o=None:acc([dict(correct=int(r['gen_choice']=='gold')) for r in pick(valid,mode=mode,**({'order':o} if o else {}))])
        out.append([m,len(g),1-len(valid)/len(g),acc(valid,'gen_agree'),gacc('READ_RULE','book_first'),gacc('APPLY_P','book_first'),gacc('READ_RULE','rule_first'),
                    f"{sum(r['gen_choice']=='p' for r in pick(valid,mode='NONAPPLY'))}/{len(pick(valid,mode='NONAPPLY'))}"])
    write_table('t6_generation',['Model','Generated items','Invalid rate','Agreement with log-prob','Gen READ_RULE bf','Gen APPLY bf','Gen READ_RULE rf','Gen leak (NONAPPLY)'],out,
                'Greedy free generation (no teacher forcing) on every 4th item; accuracies over valid parses.')

def t9_framing(rs):
    if not rs:return
    out=[]
    for f in ('neutral','default','filler'):
        na=pick(rs,mode='NONAPPLY',framing=f)
        out.append([f,leak_str(na),acc(na,'leakage'),acc(pick(rs,mode='APPLY_P',framing=f)),acc(pick(rs,mode='READ_RULE',framing=f))])
    d=paired(pick(rs,mode='NONAPPLY',framing='default'),pick(rs,mode='NONAPPLY',framing='filler'),key='leakage',match=('bundle_id','split','mode','dis'),cluster=('bundle_id','split'))
    n=paired(pick(rs,mode='NONAPPLY',framing='neutral'),pick(rs,mode='NONAPPLY',framing='filler'),key='leakage',match=('bundle_id','split','mode','dis'),cluster=('bundle_id','split'))
    write_table('t9_framing',['Framing','Leak','Leak rate','APPLY','READ_RULE'],out,
                f'Phi-3.5-mini, codebook first, fresh data. default - filler leak: {ci(d)}; neutral - filler: {ci(n)}.')

# ---------------------------------------------------------------- figures
def style(ax,title):
    ax.set_facecolor(SURFACE);ax.set_title(title,loc='left',fontsize=9,color=INK)
    for s in ('top','right'):ax.spines[s].set_visible(False)
    for s in ('left','bottom'):ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2,labelsize=8);ax.yaxis.grid(True,color=GRID,linewidth=.6);ax.set_axisbelow(True)

def grouped(ax,groups,series,values,labels,ylim=(0,1.05),fmt='{:.2f}'):
    import numpy as np
    k=len(series);w=.8/k;x=np.arange(len(groups))
    for i,_ in enumerate(series):
        ys=[values[g][i] for g in groups]
        bars=ax.bar(x+(i-(k-1)/2)*w,ys,w*.92,color=SLOTS[i],label=labels[i],edgecolor=SURFACE,linewidth=1)
        for b,y in zip(bars,ys):ax.text(b.get_x()+b.get_width()/2,y+.012*ylim[1],fmt.format(y),ha='center',va='bottom',fontsize=6.5,color=INK2)
    ax.set_xticks(x,groups,fontsize=8);ax.set_ylim(*ylim)

def legend_below(ax,ncol,drop=.13):
    ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-drop),ncol=ncol)

def save(fig,name):
    for e in ('pdf','png'):fig.savefig(FIGS/f'{name}.{e}',dpi=200)


# ---------------------------------------------------------------- round 4
def _avg_rand(rows,prefix,mode):
    """Average the random-k draws per item so they pair with the single top-k run."""
    avg=defaultdict(list)
    for r in rows:
        if r['variant'].startswith(prefix) and r['mode']==mode:avg[(r['bundle_id'],r['split'],r['mode'])].append(r['correct'])
    return [dict(bundle_id=b,split=sp,mode=md,correct=sum(v)/len(v)) for (b,sp,md),v in avg.items()]

def t10_circuit(e4c,name='t10_circuit'):
    out=[]
    for m,(rs,path) in e4c.items():
        if not rs:continue
        line=[m]
        for mode in ('READ_RULE','APPLY_P'):
            cl=pick(rs,variant='clean',mode=mode);bl=pick(rs,variant='blocked',mode=mode)
            suff=[r for r in rs if r['variant'].startswith('only_top') and r['mode']==mode]
            line+= [f'{acc(cl):.2f} / {acc(suff):.2f} / {acc(bl):.2f}',ci(paired(suff,cl))]
        for mode in ('READ_RULE','APPLY_P'):
            top=[r for r in rs if r['variant'].startswith('read_top') and r['mode']==mode]
            line.append(f'{acc(top):.2f} vs {acc(_avg_rand(rs,"read_rand",mode)):.2f}: {ci(paired(top,_avg_rand(rs,"read_rand",mode)))}')
        if path:
            clean=[dict(r,correct=r['reader_mass']) for r in path if r['variant']=='clean']
            top=[dict(r,correct=r['reader_mass']) for r in path if r['variant'].startswith('top')]
            rnd=defaultdict(list)
            for r in path:
                if r['variant'].startswith('rand'):rnd[(r['bundle_id'],r['split'],r['mode'])].append(r['reader_mass'])
            rr=[dict(bundle_id=b,split=sp,mode=md,correct=sum(v)/len(v)) for (b,sp,md),v in rnd.items()]
            line.append(f'clean {acc(clean):.3f}, top {acc(top):.3f}, rand {acc(rr):.3f}; top-rand {ci(paired(top,rr))}')
        out.append(line)
    write_table(name,['Model','READ clean / only-top-k binding / all blocked','only-top-k - clean (READ)','APPLY clean / only-top-k / all blocked',
                               'only-top-k - clean (APPLY)','Reader top-8 vs random-8 (READ)','Reader top-8 vs random-8 (APPLY)','Reader attention to queried entry under binding-head block'],out,
                'Codes, codebook first. only-top-k: every head except the top-k binding heads is blocked from the codebook (sufficiency). Reader heads: answer position -> rule tokens. Path: mean attention of the top-8 reader heads to the queried rule entry (minus other entries).')

def _jaccard(a,b):
    a,b=set(map(tuple,a)),set(map(tuple,b));return len(a&b)/len(a|b) if a|b else float('nan')

def t11_format(e4f,heads):
    out=[];curves={}
    for key,rs in e4f.items():
        if not rs:continue
        layers=sorted({r['restore_layer'] for r in rs if r['variant'].startswith('restore_L')})
        rest={mode:[acc(pick(rs,variant=f'restore_L{l}',mode=mode)) for l in layers] for mode in ('READ_RULE','APPLY_P')}
        best=max(range(len(layers)),key=lambda i:rest['READ_RULE'][i]) if layers else None
        curves[key]=dict(layers=layers,**rest,clean={md:acc(pick(rs,variant='clean',mode=md)) for md in ('READ_RULE','APPLY_P')},
                         blocked={md:acc(pick(rs,variant='blocked',mode=md)) for md in ('READ_RULE','APPLY_P')})
        top=pick(rs,variant='top8',mode='READ_RULE')
        out.append([key,curves[key]['clean']['READ_RULE'],curves[key]['blocked']['READ_RULE'],
                    f"{rest['READ_RULE'][best]:.2f} at L={layers[best]}" if best is not None else 'n/a',
                    curves[key]['clean']['APPLY_P'],curves[key]['blocked']['APPLY_P'],max(rest['APPLY_P']) if layers else float('nan'),
                    f"{acc(top):.2f} vs {acc(_avg_rand(rs,'rand8_','READ_RULE')):.2f}: {ci(paired(top,_avg_rand(rs,'rand8_','READ_RULE')))}"])
    jac=[]
    rng=random.Random(0)
    for a,b in [(x,y) for x in heads for y in heads if x<y and x.split('|')[0]==y.split('|')[0]]:
        ha,hb=heads[a],heads[b]
        if not ha or not hb:continue
        k=min(32,len(ha['top']),len(hb['top']));pool=[(l,h) for l in range(ha['depth']) for h in range(len(ha['scores'][0]))]
        rand=sum(_jaccard(rng.sample(pool,k),rng.sample(pool,k)) for _ in range(500))/500
        jac.append([a,b,_jaccard(ha['top'][:k],hb['top'][:k]),rand])
    write_table('t11_format_mechanism',['Model|format|order','READ clean','READ blocked','READ best restored','APPLY clean','APPLY blocked','APPLY best restored','Binding top-8 vs random-8 (READ)'],out,
                'Later source blocked from the earlier one in all layers; restoration writes back clean later-source states after layer L.')
    write_table('t11b_head_overlap',['Run A','Run B','Jaccard top-32 binding heads','Random-set Jaccard'],jac)
    return curves

def prediction_table(frozen,mech,name,title):
    out=[]
    for m,rs in frozen.items():
        if not rs:continue
        fixed=('bundle_id','split','framing','distractor','control')
        a=paired(pick(rs,mode='READ_RULE',order='book_first'),pick(rs,mode='APPLY_P',order='book_first'),match=fixed+('order',))
        b=paired(pick(rs,mode='READ_RULE',order='book_first'),pick(rs,mode='READ_RULE',order='rule_first'),match=fixed+('mode',))
        mk=mech.get(m) or []
        c=paired(pick(mk,variant='M1',mode='READ_RULE'),pick(mk,variant='M2',mode='READ_RULE')) if mk else (float('nan'),)*3
        out.append([m,ci(a),ci(b),ci(c),acc(pick(rs,mode='READ_RULE',order='book_first')),leak_str(pick(rs,mode='NONAPPLY'))])
    write_table(name,['Model','READ_RULE bf - APPLY bf','READ_RULE bf - rf','Knockout M1 - M2 (READ_RULE)','READ_RULE bf','Leak (all NONAPPLY)'],out,title)


def format_contrasts(frozen_sets):
    """Descriptive post-hoc summary with fully condition-matched intervals."""
    out=[]
    for style,models in frozen_sets.items():
        for model,rs in models.items():
            if not rs:continue
            left=pick(rs,mode='READ_RULE',order='book_first');right=pick(rs,mode='READ_RULE',order='rule_first')
            match=('bundle_id','split','framing','distractor','control','mode')
            contrast=paired(left,right,match=match)
            out.append([style,model,acc(left),acc(right),ci(contrast),len(left),len({(r['bundle_id'],r['split']) for r in left})])
    write_table('t21_format_order',['Format','Model','READ bf','READ rf','bf - rf [CI95]','Items per order','Bundle clusters'],out,
                'Post-hoc summary of existing probes; fixed framing, distractor, control and mode are matched exactly. '
                'Intervals containing zero are not evidence of equivalence. Formats also differ in vocabulary and wording.')

def figures(frozen,mech,adapt,curves):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURFACE})
    models=[m for m in frozen if frozen[m]]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),constrained_layout=True)
    for ax,mode,title in ((axes[0],'APPLY_P','Apply rule (two hops)'),(axes[1],'READ_RULE','Read rule (one hop, decoded value given)')):
        vals={m:[acc(pick(frozen[m],mode=mode,order='book_first')),acc(pick(frozen[m],mode=mode,order='rule_first'))] for m in models}
        grouped(ax,models,['bf','rf'],vals,['codebook first','rule first']);style(ax,title);ax.set_ylabel('accuracy',fontsize=8,color=INK2);legend_below(ax,2)
    save(fig,'f1_order_asymmetry');plt.close(fig)
    mm=[m for m in mech if mech[m]]
    fig,ax=plt.subplots(figsize=(1.3*len(mm)+1.6,2.9),constrained_layout=True)
    vals={m:[acc(pick(mech[m],variant=v,mode='READ_RULE')) for v in ('M0','M2','M1')] for m in mm}
    grouped(ax,mm,['M0','M2','M1'],vals,['no block','sham block','rule tokens blocked from codebook'])
    style(ax,'Read rule (codebook first) under attention knockout');ax.set_ylabel('accuracy',fontsize=8,color=INK2);legend_below(ax,3)
    save(fig,'f2_knockout');plt.close(fig)
    if adapt:
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
        ms=['LORA_QKVO','SEPARABLE','VECTOR_PAIR'];names=[LABEL[m] for m in ms]
        vals={LABEL[m]:[acc(pick([r for r in adapt if r['stage'].startswith(p) and r['method']==m],mode='APPLY_P',order='rule_first')) for p in ('bf_s','both_s')] for m in ms}
        grouped(axes[0],names,['bf','both'],vals,['trained codebook-first only','trained on both orders'])
        full=acc(pick([r for r in adapt if r['stage']=='full'],mode='APPLY_P',order='rule_first'))
        axes[0].axhline(full,color=INK2,linewidth=1,linestyle='--');axes[0].text(-.45,full+.035,f'frozen {full:.2f}',fontsize=7,color=INK2,ha='left')
        style(axes[0],'Apply rule, rule-first order (held out)');legend_below(axes[0],2)
        cfgs=[('Qwen 1L\nlr 2e-3','q1l_s','LORA_V'),('Qwen 4L\nlr 5e-4','bf_s','LORA_QKVO'),('Qwen 4L\nlr 2e-3','bfhi_s','LORA_QKVO'),('Phi 4L\nlr 5e-4','phi_s','LORA_QKVO')]
        lv={label:[acc(pick([r for r in adapt if r['stage'].startswith(p) and r['method']==m],mode='NONAPPLY'),'leakage') for m in (lora,'SEPARABLE','VECTOR_PAIR')]
            for label,p,lora in cfgs}
        top=max(max(v) for v in lv.values());grouped(axes[1],list(lv),['l','s','v'],lv,['LoRA','Separable','Vector pair'],ylim=(0,max(.05,top*1.25)),fmt='{:.1%}')
        style(axes[1],'Out-of-scope application after adaptation');axes[1].set_ylabel('leak rate (NONAPPLY)',fontsize=8,color=INK2)
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0,decimals=0));legend_below(axes[1],3,drop=.24)
        save(fig,'f3_adaptation');plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,len(curves),figsize=(3.6*len(curves),2.8),constrained_layout=True,squeeze=False)
        for ax,(m,c) in zip(axes[0],curves.items()):
            for i,(mode,name) in enumerate((('APPLY_P','apply'),('READ_RULE','read rule'))):
                ax.plot(c['layers'],c[mode],color=SLOTS[i],linewidth=2,marker='o',markersize=3.5,label=f'{name} (restored)')
                ax.axhline(c['clean'][mode],color=SLOTS[i],linewidth=1,linestyle='--');ax.axhline(c['blocked'][mode],color=SLOTS[i],linewidth=1,linestyle=':')
            style(ax,f'{m}: restore clean rule states after layer L');ax.set_xlabel('layer L',fontsize=8,color=INK2);ax.set_ylim(0,1.05);legend_below(ax,2,drop=.22)
        axes[0][0].set_ylabel('accuracy (dashed = clean, dotted = blocked)',fontsize=7,color=INK2)
        save(fig,'f4_restoration');plt.close(fig)


def figures_r4(fcurves,frozen_sets):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURFACE})
    models=sorted({k.split('|')[0] for k in fcurves})
    if models:
        fig,axes=plt.subplots(1,len(models),figsize=(3.7*len(models),2.9),constrained_layout=True,squeeze=False)
        for ax,m in zip(axes[0],models):
            for i,(fmt,order) in enumerate((('codes','book_first'),('codes','rule_first'),('nl','book_first'),('nl','rule_first'))):
                c=fcurves.get(f'{m}|{fmt}|{order}')
                if not c:continue
                ax.plot(c['layers'],c['READ_RULE'],color=SLOTS[i],linewidth=2,marker='o',markersize=3,label=f'{fmt}, {"codebook" if order=="book_first" else "rule"} first')
            style(ax,f'{m}: read rule after restoring at L');ax.set_xlabel('layer L',fontsize=8,color=INK2);ax.set_ylim(0,1.05);legend_below(ax,2,drop=.22)
        save(fig,'f5_format_restoration');plt.close(fig)
    rows=[]
    for fmt,frozen in frozen_sets.items():
        for m,rs in frozen.items():
            if rs:rows.append((fmt,m,acc(pick(rs,mode='READ_RULE',order='book_first'))-acc(pick(rs,mode='READ_RULE',order='rule_first'))))
    if rows:
        fmts=list(frozen_sets);ms=sorted({r[1] for r in rows})
        fig,ax=plt.subplots(figsize=(7.2,3.0),constrained_layout=True)
        import numpy as np
        x=np.arange(len(ms));w=.8/len(fmts)
        for i,f in enumerate(fmts):
            ys=[next((r[2] for r in rows if r[0]==f and r[1]==m),np.nan) for m in ms]
            bars=ax.bar(x+(i-(len(fmts)-1)/2)*w,ys,w*.92,color=SLOTS[i],label=f,edgecolor=SURFACE)
            for b_,y in zip(bars,ys):
                if y==y and abs(y)>=.08:ax.text(b_.get_x()+b_.get_width()/2,y+(.01 if y>=0 else -.04),f'{y:+.2f}',ha='center',fontsize=6,color=INK2)
        lo=min(min(r[2] for r in rows),0);ax.set_ylim(lo-.08,max(r[2] for r in rows)+.06)
        ax.axhline(0,color=INK2,linewidth=.8);ax.set_xticks(x,ms,fontsize=7.5);style(ax,'Order asymmetry: read rule, codebook-first minus rule-first (labels where |diff| >= 0.08)')
        ax.set_ylabel('accuracy difference',fontsize=8,color=INK2);legend_below(ax,len(fmts))
        save(fig,'f6_format_asymmetry');plt.close(fig)

# ---------------------------------------------------------------- round 5
def _rec(x,clean,blocked):return (x-blocked)/(clean-blocked) if clean-blocked>.05 else float('nan')

def t14_sender(path):
    grid={'READ_RULE':[],'APPLY_P':[]};con=[];fig={}
    cols=['clean','blocked','dec_all','dec_attn','dec_mlp','dec_top8','dec_top32','dec_top128','dec_rand32_','dec_rand128_','dec_top32_mlp']
    for run,rs in path.items():
        if not any(r['variant']=='dec_all' for r in rs):continue
        get=lambda v,mode:_avg_rand(rs,v,mode) if v.endswith('_') else pick(rs,variant=v,mode=mode)
        for mode in grid:grid[mode].append([run]+[acc(get(v,mode)) for v in cols])
        R='READ_RULE';cl=acc(get('clean',R));bl=acc(get('blocked',R))
        con.append([run,ci(paired(get('dec_all',R),get('clean',R))),ci(paired(get('dec_attn',R),get('dec_mlp',R))),ci(paired(get('dec_top32',R),get('dec_rand32_',R))),
                    ci(paired(get('dec_top32_mlp',R),get('dec_top32',R))),ci(paired(get('dec_top128',R),get('dec_rand128_',R))),
                    ' / '.join(f'{_rec(acc(get(v,R)),cl,bl):.2f}' for v in ('dec_attn','dec_mlp','dec_top32','dec_top128'))])
        fig[run]=[_rec(acc(get(v,R)),cl,bl) for v in ('dec_attn','dec_mlp','dec_top32','dec_rand32_')]
    head=['Run','clean','blocked','all','attn','MLP','top-8 heads','top-32','top-128','random-32','random-128','top-32 + MLP']
    note=('Rule tokens blocked from the codebook in every layer; after layer L* the clean-minus-blocked direct contributions of the listed components '
          '(layers 0..L*) are added back at the rule tokens. Random sets: mean of 3 draws. Heads ranked on held-out bundles.')
    write_table('t14a_sender_read',head,grid['READ_RULE'],'READ_RULE accuracy. '+note)
    write_table('t14b_sender_apply',head,grid['APPLY_P'],'APPLY_P accuracy. '+note)
    write_table('t14c_sender_predictions',['Run','P5a all - clean','P5b attn - MLP','P5c top-32 - random-32','P5d top-32+MLP - top-32','top-128 - random-128',
                                           'Recovered fraction: attn / MLP / top-32 / top-128'],con,'READ_RULE; paired by item, bootstrap over bundles.')
    return fig

def t15_receiver(path):
    out=[];fig={}
    for run,rs in path.items():
        if not rs:continue
        k=next((r['variant'][7:] for r in rs if r['variant'].startswith('kv_read')),None)
        if k is None:continue
        line=[run]
        for mode in ('APPLY_P','READ_RULE'):
            g=lambda v:pick(rs,variant=v,mode=mode);rnd=_avg_rand(rs,f'kv_rand{k}_',mode)
            line+=[f"{acc(g('clean')):.2f} / {acc(g('blocked')):.2f}",f"{acc(g(f'kv_read{k}')):.2f} / {acc(g(f'k_read{k}')):.2f} / {acc(g(f'v_read{k}')):.2f} / {acc(rnd):.2f}",
                   ci(paired(g(f'kv_read{k}'),rnd))]
            fig.setdefault(run,{})[mode]=[acc(g(x)) for x in (f'kv_read{k}',f'k_read{k}',f'v_read{k}')]+[acc(rnd)]
            if mode=='APPLY_P':line.append(ci(paired(g(f'k_read{k}'),g(f'v_read{k}'))))
        groups=[r.get('kv_groups') for r in rs if r['variant']==f'kv_read{k}'];line.append(str(groups[0]) if groups else '')
        read=lambda v:pick(rs,variant=v,mode='READ_RULE')
        line.extend([ci(paired(read(f'k_read{k}'),read(f'v_read{k}'))),
                     ci(paired(read(f'kv_read{k}'),read(f'k_read{k}'))),
                     ci(paired(read(f'v_read{k}'),read('blocked')))])
        out.append(line)
    write_table('t15_receiver',['Run','APPLY clean / blocked','APPLY K+V / K / V / random K+V','P6a K+V - random (APPLY)','P6b K - V (APPLY)',
                                'READ clean / blocked','READ K+V / K / V / random K+V','K+V - random (READ)','KV groups restored',
                                'K only - V only (READ)','K+V - K only (READ)','V only - blocked (READ)'],
                out,
                'Blocked run (rule tokens blocked from the codebook in every layer); the top-8 reader heads\' key/value groups get their clean keys and/or values '
                'at the rule tokens. Random: 8 random heads\' groups, mean of 3 draws.')
    return fig

def t16_route(path,mass):
    out=[]
    for run,rs in path.items():
        ms=mass.get(run) or []
        if not rs or not any(r['variant']=='blocked_vals' for r in rs):continue
        R='READ_RULE';g=lambda v:pick(rs,variant=v,mode=R);top=next(r['variant'] for r in rs if r['variant'].startswith('bind_top'));k=top[8:]
        m=lambda v:[dict(r,correct=r['reader_mass']) for r in ms if r['variant']==v and r['mode']==R]
        rm=defaultdict(list)
        for r in ms:
            if r['variant'].startswith(f'bind_rand{k}_') and r['mode']==R:rm[(r['bundle_id'],r['split'],r['mode'])].append(r['reader_mass'])
        mr=[dict(bundle_id=b,split=s,mode=md,correct=sum(v)/len(v)) for (b,s,md),v in rm.items()]
        cm=acc(m('clean'))
        out.append([run,' / '.join(f'{acc(x):.2f}' for x in (g('clean'),g('blocked'),g('blocked_vals'),g(top),_avg_rand(rs,f'bind_rand{k}_',R))),
                    ci(paired(g(top),g('blocked'))),
                    ' / '.join(f'{acc(x):.3f}' for x in (m('clean'),m('blocked'),m('blocked_vals'),m(top),mr)),
                    f"{acc(m('blocked'))/cm:.2f}" if cm>0 else 'n/a',ci(paired(m(top),m('blocked')))])
    write_table('t16_route',['Run','READ: clean / whole codebook / value tokens / top-32 binding / random-32','P7a top-32 - whole (READ)',
                             'Reader attention to queried entry: clean / whole / values / top-32 / random-32','P7b whole / clean','P7c top-32 - whole (attention)'],out,
                'READ_RULE. Blocks from the rule tokens: whole codebook (every head), codebook value tokens (every head), value tokens in the top-32 '
                'binding heads only, or 32 random heads (mean of 3). Attention: top-8 reader heads, queried entry minus the mean of other entries.')

def t19_heads(heads):
    out=[];rep=[];receiver=[];fig={}
    for run,rs in heads.items():
        if not any(r['variant'].startswith('kh_') for r in rs):continue
        R='READ_RULE';g=lambda v,m=R:pick(rs,variant=v,mode=m)
        cl,bl,grp=acc(g('clean')),acc(g('blocked')),acc(g('k_read8'))
        ks=sorted(int(v[7:]) for v in {r['variant'] for r in rs} if v.startswith('kh_read'))
        rnd=_avg_rand(rs,'kh_rand8_',R);rest=g('kh_rest8')
        frac=(acc(g('kh_read8'))-bl)/(grp-bl) if grp-bl>.05 else float('nan')
        half=next((k for k in ks if grp-bl>.05 and acc(g(f'kh_read{k}'))-bl>=(grp-bl)/2),None)
        out.append([run,f'{cl:.2f} / {bl:.2f}',grp,' / '.join(f'{acc(g(f"kh_read{k}")):.2f}' for k in ks),acc(rest) if rest else float('nan'),acc(rnd),
                    ci(paired(g('kh_read8'),rnd)),ci(paired(g('kh_read8'),rest)) if rest else 'n/a (one head per group)',f'{frac:.2f}',str(half)])
        rep.append([run,ci(paired(g('k_read8'),g('v_read8'))),ci(paired(g('kv_read8'),_avg_rand(rs,'kv_rand8_',R))),
                    ci(paired(g('kv_read8','APPLY_P'),_avg_rand(rs,'kv_rand8_','APPLY_P')))])
        receiver.append([run,cl,bl,acc(g('kv_read8')),grp,acc(g('v_read8')),acc(_avg_rand(rs,'kv_rand8_',R)),
                         ci(paired(g('k_read8'),g('v_read8'))),len(g('clean'))])
        fig[run]=dict(ks=ks,curve=[acc(g(f'kh_read{k}')) for k in ks],clean=cl,blocked=bl,group=grp,rand=acc(rnd))
    write_table('t19_heads',['Run','READ clean / blocked','group keys (k_read8)','reader heads only, k = '+'/'.join(['1','2','4','8','16']),'group-mates of top-8',
                             'random 8 heads','P9a top-8 - random-8','top-8 - group-mates','P9b top-8 / group recovery','P9c smallest k reaching half'],out,
                'READ_RULE in the blocked run with clean keys at the rule tokens for single query heads only (their group-mates keep blocked keys). '
                'Fresh bundles (seed 61); heads ranked on held-out bundles (seed 73).')
    write_table('t19b_receiver_replication',['Run','P10a K only - V only (READ)','P10b K+V readers - random (READ)','K+V readers - random (APPLY)'],rep,
                'Round-5 receiver variants on fresh bundles (seed 61).')
    write_table('t22_receiver_fresh',['Run','READ clean','READ blocked','READ K+V','READ K only','READ V only','READ random K+V',
                                    'K only - V only [CI95]','Items per variant'],receiver,
                'Fresh-bundle receiver replication (seed 61); head selection seed 73. READ_RULE, neutral framing, codebook first, no distractor. '
                'Each codes model has 48 bundle clusters.')
    return fig

def figures_r6(hfig):
    if not hfig:return
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURFACE})
    runs=[r for r in hfig if r.startswith('codes')];n=len(runs)
    if not n:return
    fig,axes=plt.subplots(1,n,figsize=(2.1*n+.6,2.8),constrained_layout=True,squeeze=False,sharey=True)
    for ax,run in zip(axes[0],runs):
        c=hfig[run]
        ax.plot(range(len(c['ks'])),c['curve'],color=SLOTS[0],linewidth=2,marker='o',markersize=3.5,label='reader heads only')
        ax.axhline(c['group'],color=SLOTS[1],linewidth=1.2,linestyle='--',label='their whole groups')
        ax.axhline(c['rand'],color=SLOTS[2],linewidth=1.2,linestyle=':',label='8 random heads')
        ax.axhline(c['blocked'],color=INK2,linewidth=.8,label='blocked')
        ax.set_xticks(range(len(c['ks'])),[str(k) for k in c['ks']],fontsize=7);style(ax,run.replace('codes_',''));ax.set_ylim(0,1.05)
        ax.set_xlabel('top-k reader heads',fontsize=7,color=INK2)
    axes[0][0].set_ylabel('read rule accuracy',fontsize=8,color=INK2)
    handles,labels=axes[0][0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7,loc='outside lower center',ncol=4)
    save(fig,'f9_heads');plt.close(fig)

def figures_r5(sender,receiver):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURFACE})
    codes={k:v for k,v in sender.items() if k.startswith('codes')}
    if codes:
        fig,ax=plt.subplots(figsize=(7.2,3.0),constrained_layout=True)
        names={k:k.replace('codes_','') for k in codes}
        vals={names[k]:[0 if x!=x else x for x in v] for k,v in codes.items()}
        grouped(ax,list(vals),['a','m','t','r'],vals,['all attention heads','all MLPs','top-32 binding heads','32 random heads'],ylim=(min(0,min(min(v) for v in vals.values()))-.05,1.15))
        style(ax,'What writes the binding: share of the read-rule gap recovered at layer L*');ax.set_ylabel('recovered fraction',fontsize=8,color=INK2)
        ax.axhline(0,color=INK2,linewidth=.8);legend_below(ax,4);save(fig,'f7_sender');plt.close(fig)
    rc={k.replace('codes_',''):v for k,v in receiver.items() if k.startswith('codes')}
    if rc:
        fig,axes=plt.subplots(1,2,figsize=(7.6,3.0),constrained_layout=True)
        for ax,mode,title in ((axes[0],'READ_RULE','Read rule'),(axes[1],'APPLY_P','Apply rule')):
            vals={k.replace('_','\n',1):v[mode] for k,v in rc.items()}
            grouped(ax,list(vals),['kv','k','v','r'],vals,['reader K+V','reader K only','reader V only','random heads K+V'],fmt='{:.1f}')
            style(ax,f'{title}: blocked run, reader K/V restored');legend_below(ax,2,drop=.28)
        axes[0].set_ylabel('accuracy',fontsize=8,color=INK2);save(fig,'f8_receiver');plt.close(fig)

def main():
    global RUNS,TABLES,FIGS
    p=argparse.ArgumentParser();p.add_argument('--layout',default=str(OUT/'layout_original.json'));p.add_argument('--runs',default=str(OUT.parent/'runs'))
    p.add_argument('--out',default=str(OUT),help='tables/ and figures/ are written here (default: paper/)')
    a=p.parse_args();RUNS=Path(a.runs);L=json.loads(Path(a.layout).read_text())
    TABLES=Path(a.out)/'tables';FIGS=Path(a.out)/'figures';TABLES.mkdir(parents=True,exist_ok=True);FIGS.mkdir(parents=True,exist_ok=True)
    frozen={m:load(s) for m,s in L.get('frozen',{}).items()}
    mech={m:load(s,'mech') for m,s in L.get('mech',{}).items()}
    t1_frozen(frozen);t2_knockout(mech);t3_e2(load(L.get('e2'),'mech'))
    adapt=load(L.get('adaptation'))
    t4_adaptation(adapt)
    e4={m:(load(s,'e4'),zip_json(s,'head_scores.json')) for m,s in L.get('e4',{}).items()};curves=t5_e4(e4)
    t6_generation({m:load(s) for m,s in L.get('generation',{}).items()})
    nl_frozen={m:load(s) for m,s in L.get('nl_probe',{}).items()};nl_mech={m:load(s,'mech') for m,s in L.get('nl_mech',{}).items()}
    if any(nl_frozen.values()):t1_frozen(nl_frozen,'t6b_nl_frozen','Natural-language variant (badge -> room directory, prose procedures), frozen models.')
    if any(nl_mech.values()):t2_knockout(nl_mech,'t6c_nl_knockout','Natural-language variant: READ_RULE under knockout.')
    big={m:load(s) for m,s in L.get('scale_probe',{}).items()};bigm={m:load(s,'mech') for m,s in L.get('scale_mech',{}).items()}
    if any(big.values()):t1_frozen(big,'t7_scale_frozen','Larger model, frozen probe (codes).')
    if any(bigm.values()):t2_knockout(bigm,'t7b_scale_knockout','Larger model: READ_RULE under knockout.')
    fr=load(L.get('framing'))
    for r in fr:r['dis']=r['condition'][1]
    t9_framing(fr)
    figures({**frozen,**big},{**mech,**bigm},adapt,curves)
    # round 4
    e4c={m:(load(s,'e4'),load(dict(s,kind='e4path'))) for m,s in L.get('e4_circuit',{}).items()}
    if any(v[0] for v in e4c.values()):t10_circuit(e4c)
    e4f={k:load(s,'e4') for k,s in L.get('e4_format',{}).items()}
    fcurves=t11_format(e4f,{k:zip_json(s,'head_scores.json') for k,s in L.get('e4_format',{}).items()}) if any(e4f.values()) else {}
    fam={m:load(s) for m,s in L.get('families_probe',{}).items()};famm={m:load(s,'mech') for m,s in L.get('families_mech',{}).items()}
    famnl={m:load(s) for m,s in L.get('families_nl',{}).items()}
    if any(fam.values()):
        t1_frozen(fam,'t12_families_frozen','New families and scale, frozen probe (codes).');t2_knockout(famm,'t12b_families_knockout','New families and scale: READ_RULE under knockout.')
        prediction_table(fam,famm,'t12c_families_predictions','Pre-registered G3 predictions (codes): each CI should exclude 0 in >= 2 of 3 models.')
        if any(famnl.values()):t1_frozen(famnl,'t12d_families_nl','New families and scale, prose variant (nl).')
    frozen_sets={'codes':{**frozen,**big,**fam},'nl':{**nl_frozen,**famnl}}
    for style in ('nl_gate','nl_shelf'):
        pr={m:load(s) for m,s in L.get(f'{style}_probe',{}).items()};mk={m:load(s,'mech') for m,s in L.get(f'{style}_mech',{}).items()}
        if any(pr.values()):
            prediction_table(pr,mk,f't13_{style}',f'Prose template {style}: pre-registered G4 predictions.');frozen_sets[style]=pr
    figures_r4(fcurves,{k:v for k,v in frozen_sets.items() if any(v.values())})
    # round 5
    path={k:load(s,'path') for k,s in {**L.get('path',{}),**L.get('path_llama',{})}.items()}
    pmass={k:load(s,'pathmass') for k,s in {**L.get('path_mass',{}),**L.get('path_llama_mass',{})}.items()}
    if any(path.values()):figures_r5(t14_sender(path),t15_receiver(path));t16_route(path,pmass)
    # round 6 (families 5 and 6 share tables t17/t18)
    hd={k:load(s,'path') for k,s in L.get('heads',{}).items()}
    if any(hd.values()):figures_r6(t19_heads(hd))
    fams=lambda key:{**L.get(f'families5_{key}',{}),**L.get(f'families6_{key}',{})}
    f5={m:load(s) for m,s in fams('probe').items()};f5m={m:load(s,'mech') for m,s in fams('mech').items()}
    f5nl={m:load(s) for m,s in fams('nl').items()}
    if any(f5.values()):
        t1_frozen(f5,'t17_families5_frozen','Round 5-6 families (Gemma-2, Mistral, Llama-3), frozen probe (codes).')
        t2_knockout(f5m,'t17b_families5_knockout','Round 5-6 families: READ_RULE under knockout.')
        prediction_table(f5,f5m,'t17c_families5_predictions','Pre-registered P8a-c (codes); weak models (READ_RULE bf < 0.6) are not counted for P8b-c.')
        if any(f5nl.values()):t1_frozen(f5nl,'t17d_families5_nl','Round 5-6 families, prose variant (nl).')
    e4c5={m:(load(s,'e4'),load(dict(s,kind='e4path'))) for m,s in {**L.get('e4_circuit5',{}),**L.get('e4_circuit6',{})}.items()}
    if any(v[0] for v in e4c5.values()):t10_circuit(e4c5,'t18_circuit_families5')
    # Manuscript-wide summaries from the existing receipts; no new model runs.
    all_frozen={**frozen,**big,**fam,**f5};all_mech={**mech,**bigm,**famm,**f5m}
    prediction_table(all_frozen,all_mech,'t20_all_models','All 13 frozen codes models; corrected matching preserves every fixed condition. '
                     'This combined table is a post-hoc summary of existing experiments, not a new registered prediction.')
    frozen_sets['codes']=all_frozen;frozen_sets['nl'].update(f5nl)
    format_contrasts(frozen_sets)
    figures_r4(fcurves,{k:v for k,v in frozen_sets.items() if any(v.values())})
    notes=sorted(set(NOTES));(TABLES/'NOTES.md').write_text('\n'.join(notes)+'\n' if notes else 'All layout sources found.\n')
    print('tables:',sorted(p.name for p in TABLES.glob('*.md')));print('figures:',sorted(p.name for p in FIGS.glob('*.png')));print('notes:',notes)

if __name__=='__main__':main()
