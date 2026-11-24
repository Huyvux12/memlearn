"""Check manuscript tables against regenerated CSVs; no inference or GPU use."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(name):
    with (ROOT/'tables'/f'{name}.csv').open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def markdown_tables(text):
    groups=[];current=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):
            cells=[x.strip() for x in line.split('|')[1:-1]]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells):current.append(cells)
        elif current:groups.append(current);current=[]
    return groups


def ci_numbers(text):return tuple(float(x) for x in re.findall(r'[+-]?\d+\.\d+',text))


def verify():
    source=ROOT/'DRAFT_v5.md';text=source.read_text(encoding='utf-8');tables=markdown_tables(text)
    assert len(tables)==7
    checks=[]
    frozen={r['Model']:r for name in ('t1_frozen','t7_scale_frozen','t12_families_frozen','t17_families5_frozen') for r in read(name)}
    assert len(frozen)==13 and len(tables[0])-1==13
    for row in tables[0][1:]:
        r=frozen[row[0]]
        expected=[f'{float(r[key]):.2f}' for key in ('READ_RULE bf','READ_RULE rf','APPLY bf','APPLY rf')]+[r['Leak bf']]
        assert row[1:]==expected,(row,expected)
    checks.append('Table 1: all 13 frozen model rows match the source tables')

    tags={'Qwen3-4B':'codes_qwen3_4b','Qwen3-8B':'codes_qwen3_8b','Phi-3.5-mini':'codes_phi35_mini',
          'Mistral-7B':'codes_mistral_7b','Gemma-2-9B':'codes_gemma2_9b','Llama-3.1-8B':'codes_llama31_8b'}
    receiver={r['Run']:r for r in read('t15_receiver')}
    for row in tables[1][1:]:
        r=receiver[tags[row[0]]]
        expected=[s.strip() for field in ('READ clean / blocked','READ K+V / K / V / random K+V') for s in r[field].split('/')]
        assert row[1:7]==expected,(row,expected)
        assert ci_numbers(row[7])==ci_numbers(r['K only - V only (READ)'])
    checks.append('Table 2: original receiver accuracies and paired K-minus-V intervals match')

    heads={r['Run']:r for r in read('t19_heads')}
    for row in tables[2][1:]:
        r=heads[tags[row[0]]]
        expected=[r['READ clean / blocked'].split('/')[1].strip(),f"{float(r['group keys (k_read8)']):.2f}",
                  r['reader heads only, k = 1/2/4/8/16'].split('/')[3].strip(),f"{float(r['group-mates of top-8']):.2f}",
                  f"{float(r['random 8 heads']):.2f}",r['P9b top-8 / group recovery']]
        assert row[1:]==expected,(row,expected)
    checks.append('Table 3: query-head accuracy, controls and recovery fractions match')

    adaptation=read('t4_adaptation')
    settings={'bf, lr 5e-4':'Qwen3-4B, 4 layers, lr 5e-4, train bf',
              'bf, lr 2e-3':'Qwen3-4B, 4 layers, lr 2e-3, train bf',
              'bf+rf, lr 5e-4':'Qwen3-4B, 4 layers, lr 5e-4, train bf+rf'}
    for row in tables[3][1:]:
        method='Vector pair' if row[1]=='Vector Pair' else row[1]
        r=next(r for r in adaptation if r['Setting']==settings[row[0]] and r['Method']==method)
        assert row[2:]==[f"{float(r['APPLY bf']):.2f}",f"{float(r['APPLY rf']):.2f}",r['Leak (NONAPPLY)']]
        assert r['Seeds']=='5'
    checks.append('Table 4: all nine adaptation rows match five-seed results')

    fresh={r['Run']:r for r in read('t22_receiver_fresh')}
    replication={r['Run']:r for r in read('t19b_receiver_replication')}
    for row in tables[5][1:]:
        r=fresh[tags[row[0]]];rep=replication[tags[row[0]]]
        assert row[1:3]==[f"{float(r['READ K only']):.2f}",f"{float(r['READ V only']):.2f}"]
        assert ci_numbers(row[3])==ci_numbers(rep['P10a K only - V only (READ)'])
        assert ci_numbers(row[4])==ci_numbers(rep['P10b K+V readers - random (READ)'])
    checks.append('Table A2: all six fresh receiver results and paired intervals match')

    summary=read('t20_all_models')
    positive_read=sum(ci_numbers(r['READ_RULE bf - APPLY bf'])[1]>0 for r in summary)
    positive_order=sum(ci_numbers(r['READ_RULE bf - rf'])[1]>0 for r in summary)
    assert (positive_read,positive_order)==(11,13)
    checks.append('Narrative counts: 11/13 resolved READ-APPLY gaps; 13/13 resolved codes order gaps')
    sender=read('t14a_sender_read')
    original=['codes_qwen3_4b','codes_qwen3_8b','codes_phi35_mini','codes_mistral_7b']
    sanity={r['Run']:float(r['clean'])-float(r['all']) for r in sender if r['Run'] in original}
    assert sum(v<=.10 for v in sanity.values())==1
    checks.append('Sender sanity: only one of the four original codes models meets the 0.10 guard')

    refs=set(map(int,re.findall(r'^(\d+)\. ',text,re.M)))
    assert refs==set(range(1,18))
    used=set()
    for group in re.findall(r'\[(\d+(?:[-,]\s*\d+)*)\]',text.split('## References')[0]):
        if '-' in group:
            lo,hi=map(int,group.split('-'));used.update(range(lo,hi+1))
        else:used.update(map(int,group.split(',')))
    assert used<=refs and used==refs
    for relative in re.findall(r'!\[.*?\]\((.*?)\)',text):assert (ROOT/relative).is_file(),relative
    assert not re.search(r'<AUTHORS>|\{\{|TODO|TBD|fourteen|14 models',text,re.I)
    checks.append('All 17 citations resolve; both manuscript figures exist; no unresolved placeholders')

    from pypdf import PdfReader
    pdf=ROOT.parent/'output'/'pdf'/'MEMLEARN_manuscript_v5.pdf';doc=PdfReader(pdf)
    extracted='\n'.join(page.extract_text() for page in doc.pages)
    assert 'Binding Makes Rules Addressable' in extracted and 'Appendix B.' in extracted
    assert 'The combined-model' in extracted
    assert all(len(page.extract_text())>300 for page in doc.pages)
    assert '\ufffd' not in extracted and '\u25a0' not in extracted
    checks.append('PDF has searchable text, complete end matter and no empty pages or replacement glyphs')
    result=dict(status='PASS',date='2026-11-17',new_gpu_runs=0,checks=checks,pdf_pages=len(doc.pages),
                manuscript_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),sender_sanity_deficits=sanity,
                source_tables=len(list((ROOT/'tables').glob('*.csv'))))
    (ROOT/'manuscript_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
