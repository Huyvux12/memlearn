"""Build the shareable reproduction package: release/MEMLEARN_REPRO_<version>/ and a zip of it.

    python tools/build_release.py

Includes code, tests, specs, pre-registrations, analysis, and a results summary (tables, figures, result-archive
hashes). Excludes raw receipts, secrets (.env), account-specific deploy folders and internal working notes.
Refuses to finish if a known personal or secret marker is found in any shipped text file.
"""
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INCLUDE_DIRS=['memlearn_attention','tests','tools','configs','docs']
INCLUDE_FILES=['README.md','LICENSE','requirements.txt','pyproject.toml','paper/analysis.py','paper/layout_original.json',
               'paper/layout_reproduce.json','paper/refs.bib','paper/DRAFT_v4.md','deploy/reproduce_all/job.json','deploy/colab_round3/job.json','deploy/colab_round4/job.json','deploy/colab_round5/job.json','deploy/colab_round6/job.json']
TEXT=('.py','.md','.json','.txt','.toml','.bib','.csv','')
# Generic secret patterns; project-specific names (accounts, e-mail domains) live in the unshipped .release_forbidden file.
GENERIC=[r'KGAT_[A-Za-z0-9]',r'KAGGLE[-_]API\s*=',r'C:\\\\?Users\\\\?',r'hf_[A-Za-z0-9]{20,}',r'"key"\s*:\s*"[0-9a-f]{32}"']

def forbidden():
    local=ROOT/'.release_forbidden'
    extra=[re.escape(x.strip()) for x in local.read_text(encoding='utf-8').splitlines() if x.strip() and not x.startswith('#')] if local.exists() else []
    return re.compile('|'.join(GENERIC+extra),re.I)

def result_hashes():
    """Result-archive hashes from deploy/EXECUTION.json, without kernel slugs or account names."""
    p=ROOT/'deploy'/'EXECUTION.json'
    if not p.exists():return '# Result hashes\n\n(not available)\n'
    runs=json.loads(p.read_text(encoding='utf-8'))['runs'];lines=['# Result archives of the original runs','',
          'Raw receipts are not shipped. These SHA256 values identify the archives that produced `results_summary/tables`.','',
          '| Phase | Platform / GPU | Source version | Result archive SHA256 | Analysis |','|:---|:---|:---|:---|:---|']
    for r in runs:
        sha=r.get('results_zip_sha256') or ' / '.join(f"{k}: {v['zip_sha256']}" for k,v in r.get('outputs',{}).items())
        if not sha:continue
        plat=r.get('gpu') or r.get('accelerator','')
        lines.append(f"| {r['phase']} | {plat} | {r.get('source_version','1.0.0')} | `{sha}` | {r.get('analysis','')} |")
    return '\n'.join(lines)+'\n'

def main():
    version=re.search(r"__version__ = '([^']+)'",(ROOT/'memlearn_attention'/'__init__.py').read_text()).group(1)
    name=f'MEMLEARN_REPRO_{version}';out=ROOT/'release'/name
    if out.exists():shutil.rmtree(out)
    for d in INCLUDE_DIRS:shutil.copytree(ROOT/d,out/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for f in INCLUDE_FILES:
        src=ROOT/f
        if not src.exists():print('skip missing',f);continue
        (out/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out/f)
    summary=out/'results_summary'
    shutil.copytree(ROOT/'paper'/'tables',summary/'tables');shutil.copytree(ROOT/'paper'/'figures',summary/'figures')
    (summary/'RESULT_HASHES.md').write_text(result_hashes(),encoding='utf-8')
    bad=[];FORBIDDEN=forbidden()
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in TEXT:
            text=p.read_text(encoding='utf-8',errors='ignore')
            for m in FORBIDDEN.finditer(text):bad.append(f'{p.relative_to(out)}: {m.group(0)}')
    if bad:
        print('REFUSING: personal/secret markers found:\n'+'\n'.join(bad[:20]));sys.exit(1)
    target=ROOT/'release'/f'{name}.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,f'{name}/{p.relative_to(out).as_posix()}')
    print('release',target,'files',sum(1 for p in out.rglob('*') if p.is_file()),'sha256',hashlib.sha256(target.read_bytes()).hexdigest())

if __name__=='__main__':main()
