"""Regression checks for the manuscript's condition-matched contrasts (no torch)."""
import importlib.util
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('paper_analysis',Path(__file__).resolve().parents[1]/'paper'/'analysis.py')
analysis=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def row(bundle,framing,mode,correct,order='book_first'):
    return dict(bundle_id=bundle,split='test_id',condition=[framing,'none',order],
                control='FULL',mode=mode,correct=correct,leakage=0)


def test_prediction_contrasts_preserve_framing_and_ignore_only_the_changed_factor(tmp_path,monkeypatch):
    # Last control framing is deliberately wrong: the old overwrite gave +0.75
    # instead of +0.25 for READ/APPLY and +0.75 instead of +0.50 for order.
    rows=[]
    for bundle in ('a','b'):
        rows.extend([row(bundle,'neutral','READ_RULE',1),row(bundle,'default','READ_RULE',1 if bundle=='a' else 0),
                     row(bundle,'neutral','APPLY_P',1),row(bundle,'default','APPLY_P',0),
                     row(bundle,'neutral','READ_RULE',1 if bundle=='a' else 0,'rule_first'),
                     row(bundle,'default','READ_RULE',0,'rule_first')])
    captured=[]
    monkeypatch.setattr(analysis,'write_table',lambda name,header,values,note='':captured.extend(values))
    analysis.prediction_table({'test':rows},{},'unused','test')
    assert captured[0][1].startswith('+0.25 ')
    assert captured[0][2].startswith('+0.50 ')
    fixed=('bundle_id','split','framing','distractor','control','order')
    left=analysis.pick(rows,mode='READ_RULE',order='book_first')
    right=analysis.pick(rows,mode='APPLY_P',order='book_first')
    assert analysis.paired(left,right,match=fixed)[0]==analysis.acc(left)-analysis.acc(right)


def test_ambiguous_pairing_fails_instead_of_overwriting_a_condition():
    left=[row('a','neutral','READ_RULE',1),row('a','default','READ_RULE',0)]
    with pytest.raises(ValueError,match='Duplicate paired key'):
        analysis.paired(left,left)


def test_order_pairing_is_invariant_to_control_row_order():
    left=[row('a','neutral','READ_RULE',1),row('a','default','READ_RULE',0)]
    right=[row('a','neutral','READ_RULE',0,'rule_first'),row('a','default','READ_RULE',1,'rule_first')]
    match=('bundle_id','split','framing','distractor','control','mode')
    assert analysis.paired(left,right,match=match)==analysis.paired(left,list(reversed(right)),match=match)
