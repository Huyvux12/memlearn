"""Benchmark v2: per-mode keys, derangement rules, distractor rule, framing pressure. See docs/BENCHMARK_V2_SPEC.md."""
import random
import re
from .data import canonical,digest,_CUES,TRAIN_FAMILIES,DEV_FAMILIES,TEST_FAMILIES

MODES=('READ_DEC','READ_RULE','APPLY_P','APPLY_Q','NONAPPLY')
CONDITIONS=tuple((f,d,o) for f in ('neutral','default') for d in ('none','q') for o in ('book_first','rule_first'))

def _derangement(values,rng,avoid=None):
    # Rejection sampling: no fixed points, and no agreement with `avoid` (Q must differ from P everywhere).
    while True:
        image=values[:];rng.shuffle(image);m=dict(zip(values,image))
        if all(m[v]!=v for v in values) and (avoid is None or all(m[v]!=avoid[v] for v in values)):return m

# Natural-language variant (external validity): badge numbers -> room names, prose procedures.
ROOMS=('Amber', 'Birch', 'Cedar', 'Coral', 'Dune', 'Elm', 'Fern', 'Flint', 'Garnet', 'Hazel', 'Indigo', 'Ivory', 'Jade', 'Juniper', 'Kelp', 'Lark', 'Linen', 'Maple', 'Marble', 'Moss', 'Nectar', 'Oak', 'Olive', 'Onyx', 'Opal', 'Pearl', 'Pine', 'Plum', 'Quartz', 'Raven', 'Reed', 'Rose', 'Ruby', 'Rust', 'Sage', 'Sand', 'Slate', 'Spruce', 'Stone', 'Teal', 'Thistle', 'Topaz', 'Umber', 'Velvet', 'Violet', 'Willow', 'Wren', 'Yew', 'Zinc', 'Ash', 'Basil', 'Beryl', 'Brook', 'Clay', 'Cloud', 'Copper', 'Crane', 'Daisy', 'Delta', 'Echo', 'Ember', 'Falcon', 'Frost', 'Gale', 'Glade', 'Granite', 'Harbor', 'Heron', 'Iris', 'Iron', 'Jasper', 'Lake', 'Laurel', 'Lotus', 'Lynx', 'Meadow', 'Mint', 'Nova', 'Orchid', 'Otter', 'Pebble', 'Poppy', 'Prairie', 'River', 'Robin', 'Saffron', 'Shore', 'Silver', 'Sky', 'Spring', 'Summit', 'Sunset', 'Tide', 'Tulip', 'Valley', 'Vine', 'Wave', 'Wheat', 'Wolf', 'Zephyr')

def make_bundle(seed,split,index,family,size=6,style='codes'):
    tag=['v2',seed,split,index,family]+([] if style=='codes' else [style])
    rng=random.Random(digest(tag))
    if style=='codes':keys=rng.sample([f'x{i:02d}' for i in range(100)],size);values=rng.sample([f'z{i:02d}' for i in range(100)],size)
    elif style.startswith('nl'):keys=rng.sample([str(i) for i in range(100,1000)],size);values=rng.sample(list(ROOMS),size)
    else:raise ValueError(style)
    p=_derangement(values,rng);q=_derangement(values,rng,avoid=p)
    # Distinct key per mode: no field can be copied from another mode's answer.
    mode_keys=dict(zip(MODES,rng.sample(keys,len(MODES))))
    other=dict(zip(MODES,[rng.random() for _ in MODES]))
    return dict(id=digest(tag)[:20],seed=seed,split=split,family=family,style=style,
                codebook=dict(zip(keys,values)),p=p,q=q,mode_keys=mode_keys,other_draw=other)

def dataset(n_test=32,n_ood=32,seed=17,style='codes'):
    fam={'test_id':TRAIN_FAMILIES,'test_ood':TEST_FAMILIES}
    return {split:[make_bundle(seed,split,i,fam[split][i%len(fam[split])],style=style) for i in range(n)] for split,n in (('test_id',n_test),('test_ood',n_ood))}

def splits(n_train=64,n_test=16,n_ood=16,seed=17,style='codes'):
    """Train bundles use the ID cue families; codebooks are seeded per split so no bundle is shared."""
    out=dataset(n_test,n_ood,seed=seed,style=style)
    out['train']=[make_bundle(seed,'train',i,TRAIN_FAMILIES[i%len(TRAIN_FAMILIES)],style=style) for i in range(n_train)]
    return out

def target(b,mode):
    """Supervised continuation; identical to the probe prefix plus the gold candidate, so training and scoring agree."""
    return prefix(b,mode)+candidates(b,mode)[0][1]+'"}'

def facts(b,mode):
    base=b['codebook'][b['mode_keys'][mode]]
    return dict(key=b['mode_keys'][mode],base=base,p=b['p'][base],q=b['q'][base])

def _other(b,mode,exclude):
    pool=sorted(v for v in b['codebook'].values() if v not in exclude)
    if not pool:raise ValueError('Codebook too small for a distinct distractor candidate')
    return pool[int(b['other_draw'][mode]*len(pool))]

def candidates(b,mode):
    """Ordered (label, value) pairs; gold first. Values are distinct by construction."""
    f=facts(b,mode)
    if mode=='READ_DEC':c=[('gold',f['base']),('input_key',f['key']),('p',f['p'])]
    elif mode=='READ_RULE':c=[('gold',f['p']),('base',f['base']),('q',f['q'])]
    elif mode=='APPLY_P':c=[('gold',f['p']),('base',f['base']),('q',f['q'])]
    elif mode=='APPLY_Q':c=[('gold',f['q']),('base',f['base']),('p',f['p'])]
    elif mode=='NONAPPLY':c=[('gold',f['base']),('p',f['p']),('q',f['q'])]
    else:raise ValueError(mode)
    c.append(('other',_other(b,mode,{v for _,v in c})))
    if len({v for _,v in c})!=len(c):raise ValueError('Candidate collision')
    return c

def prefix(b,mode):
    """Teacher-forced JSON prefix; the candidate value and '"}' follow it."""
    if mode=='READ_DEC':return '{"decoded_value":"'
    if mode=='READ_RULE':return '{"decoded_value":"'+facts(b,mode)['base']+'","rule_result":"'
    return '{"answer":"'

CONTRACT=('You solve exact codebook queries. Decode an input key using CODEBOOK: the decoded value is the CODEBOOK value '
          'mapped from INPUT_KEY, never the input key itself. A procedure replaces the decoded value using its RULE_MAP. '
          'The request states which procedure, if any, to apply. READ asks for information without applying anything. '
          'Return only one JSON object, no explanation or markdown. READ schema: {"decoded_value":"...","rule_result":"..."} '
          '(rule_result only when asked). Other requests: {"answer":"..."}.')

DEFAULT_FRAMING='STANDARD PROCEDURE P: in this workflow every decoded value is normally passed through P before it is reported.'
# Same slot and similar token count as DEFAULT_FRAMING, but no instruction content (Phi framing control).
FILLER_FRAMING='ARCHIVE NOTE: the entries in the tables below were copied from the original ledger without any changes.'

def _contract(desc,noun,key,source,thing):
    return (f'You answer questions about {desc}. The {noun} of a {key} is the {thing} listed for it '
            f'in the {source}, never the {key} number. A procedure moves {"an" if noun[0] in "aeiou" else "a"} {noun} to another {thing} according to its list. '
            'The question states which procedure, if any, to apply. A READ question asks for information without applying anything. '
            f'Return only one JSON object with {thing} names, no explanation or markdown. READ schema: {{"decoded_value":"...","rule_result":"..."}} '
            '(rule_result only when asked). Other questions: {"answer":"..."}.')

# Prose templates (external validity). 'nl' is byte-identical to v2.0.0; the others vary the relational frame and vocabulary.
NL_TEMPLATES={
 'nl':dict(book_head='STAFF DIRECTORY: ',book_entry='badge {k} is assigned to room {v}',
           rule_head={'P':'PROCEDURE P (relocation): ','Q':'PROCEDURE Q (renovation): '},rule_entry='room {v} moves to room {w}',
           input='BADGE: ',noun='assigned room',lookup='directory lookup',
           read_dec='QUESTION (READ): Which room is this badge assigned to? Give only the assigned room.',
           read_rule='QUESTION (READ): Give the assigned room and, hypothetically, the room it would move to under procedure P, without applying P to the answer.',
           contract=_contract('a staff directory and room procedures','assigned room','badge','STAFF DIRECTORY','room')),
 'nl_gate':dict(book_head='FLIGHT BOARD: ',book_entry='flight {k} departs from gate {v}',
           rule_head={'P':'GATE CHANGE P (reassignment): ','Q':'GATE CHANGE Q (maintenance): '},rule_entry='gate {v} is replaced by gate {w}',
           input='FLIGHT: ',noun='listed gate',lookup='board lookup',
           read_dec='QUESTION (READ): Which gate is this flight listed at? Give only the listed gate.',
           read_rule='QUESTION (READ): Give the listed gate and, hypothetically, the gate it would become under gate change P, without applying P to the answer.',
           contract=_contract('a flight board and gate changes','listed gate','flight','FLIGHT BOARD','gate')),
 'nl_shelf':dict(book_head='INVENTORY: ',book_entry='item {k} is stored on shelf {v}',
           rule_head={'P':'TRANSFER P (restock): ','Q':'TRANSFER Q (audit): '},rule_entry='shelf {v} is moved to shelf {w}',
           input='ITEM: ',noun='stored shelf',lookup='inventory lookup',
           read_dec='QUESTION (READ): Which shelf is this item stored on? Give only the stored shelf.',
           read_rule='QUESTION (READ): Give the stored shelf and, hypothetically, the shelf it would be moved to under transfer P, without applying P to the answer.',
           contract=_contract('an inventory and shelf transfers','stored shelf','item','INVENTORY','shelf')),
}
NL_CONTRACT=NL_TEMPLATES['nl']['contract']

def span_patterns(style):
    """Regexes for the codebook line, the P rule line, a codebook entry (key, value) and a rule entry (left, right)."""
    if style in (None,'codes'):
        return dict(book=r'CODEBOOK: \{[^\n]*\}',rule=r'RULE_MAP for [PQ]: \{[^\n]*\}',rule_p=r'RULE_MAP for P: \{[^\n]*\}',
                    book_entry=r'"(x\d\d)":"(z\d\d)"',rule_entry=r'"(z\d\d)":"(z\d\d)"')
    t=NL_TEMPLATES[style];esc=lambda x:re.escape(x)
    ent=lambda f,a,b:esc(f).replace(esc('{k}') if '{k}' in f else esc('{v}'),a,1).replace(esc('{v}') if '{k}' in f else esc('{w}'),b,1)
    return dict(book=esc(t['book_head'])+r'[^\n]*',rule='(?:'+'|'.join(esc(h) for h in t['rule_head'].values())+r')[^\n]*',
                rule_p=esc(t['rule_head']['P'])+r'[^\n]*',book_entry=ent(t['book_entry'],r'(\d+)',r'(\w+)'),rule_entry=ent(t['rule_entry'],r'(\w+)',r'(\w+)'))

def _nl(text,t):
    return text.replace('the decoded value','the '+t['noun']).replace('decoded value',t['noun']).replace('decoding stage',t['lookup'])

def render(b,mode,condition,control='FULL'):
    framing,distractor,order=condition
    t=NL_TEMPLATES.get(b.get('style'))
    if t:
        book=t['book_head']+'; '.join(t['book_entry'].format(k=k,v=v) for k,v in b['codebook'].items())+'.'
        rules=[t['rule_head']['P']+'; '.join(t['rule_entry'].format(v=v,w=w) for v,w in b['p'].items())+'.']
    else:
        book='CODEBOOK: '+canonical(b['codebook'])
        rules=['RULE_MAP for P: '+canonical(b['p'])]
    if framing=='default':rules.insert(0,DEFAULT_FRAMING)
    elif framing=='filler':rules.insert(0,FILLER_FRAMING)
    elif framing!='neutral':raise ValueError(framing)
    if distractor=='q':
        rules.append(t['rule_head']['Q']+'; '.join(t['rule_entry'].format(v=v,w=w) for v,w in b['q'].items())+'.' if t
                     else 'RULE_MAP for Q: '+canonical(b['q']))
    if order=='book_first':parts=[book]+rules
    elif order=='rule_first':parts=rules+[book]
    elif order=='rule_first_bookrepeat':parts=rules+[book,book]  # mechanism probe: codebook again, nearest the answer
    else:raise ValueError(order)
    if control=='SCOPE_PROMPT':
        parts.insert(0,'Use each procedure only when the request authorizes it. Knowing a procedure does not mean every output applies it.')
    text='\n'.join(parts)+('\n'+t['input'] if t else '\nINPUT_KEY: ')+b['mode_keys'][mode]+'\n'
    yes,no=_CUES[b['family']]
    if t:yes,no=_nl(yes,t),_nl(no,t)
    if mode=='READ_DEC':text+=t['read_dec'] if t else 'REQUEST READ: Retrieve only the decoded value.'
    elif mode=='READ_RULE':text+=t['read_rule'] if t else 'REQUEST READ: Retrieve the decoded value and the hypothetical rule_result of P, without applying P to the answer.'
    elif mode=='APPLY_P':text+='REQUEST: '+yes
    elif mode=='APPLY_Q':text+='REQUEST: '+re.sub(r'\bP\b','Q',yes)
    elif mode=='NONAPPLY':text+='REQUEST: '+no
    else:raise ValueError(mode)
    return text

def modes_for(condition):
    return [m for m in MODES if m!='APPLY_Q' or condition[1]=='q']

def items(ds,controls=('FULL',),conditions=CONDITIONS,modes=MODES):
    out=[]
    for split,bundles in ds.items():
        for b in bundles:
            for condition in conditions:
                for mode in [m for m in modes_for(condition) if m in modes]:
                    for control in controls:
                        out.append(dict(id=digest([b['id'],mode,condition,control])[:24],bundle=b,bundle_id=b['id'],split=split,
                                        family=b['family'],mode=mode,condition=condition,control=control))
    return out

def messages(item):
    t=NL_TEMPLATES.get(item['bundle'].get('style'));contract=t['contract'] if t else CONTRACT
    return [{'role':'system','content':contract},{'role':'user','content':render(item['bundle'],item['mode'],item['condition'],item['control'])}]
