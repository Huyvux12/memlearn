# Verification record — 2026-10-17

## Manuscript analysis revision — 2026-11-17

The current paper is `paper/DRAFT_v5.md`. This revision uses existing receipts only and performs no new pretrained inference or training. Three CPU-only regression tests in `tests/test_paper_analysis.py` pass, covering framing-preserving contrasts, duplicate matching-key rejection and order-independent control matching. The corrected analysis rebuilds 36 tables from all layout sources without missing inputs. Receiver READ diagnostics and three combined summaries are computed from existing runs.

The PDF is built with `paper/build_manuscript.py`; `paper/verify_manuscript.py` checks its four main tables against analysis CSVs, statistical counts, sender sanity-guard outcomes, reference coverage and PDF text completeness. Results are written to `paper/manuscript_validation.json`. All PDF pages are rendered and visually reviewed separately. These checks do not rerun the original GPU experiments.

The earlier whole-suite Windows audit yielded 121 passed and 7 failed. Five encoding-related failures passed when rerun with `python -X utf8`; two process-supervisor tests require Unix `os.killpg`. This is not a claim of a new full-suite pass or Linux rerun for the manuscript revision. The historical 56-test record below describes the older build.

## Historical build verification

CPU runtime: Python 3.12, PyTorch 2.6.0+cpu, transformers 4.57.1. No CUDA GPU is available in the build environment. No pretrained Qwen4B experimental result is claimed.

Full pytest suite: **56 passed**. It exercises real tiny randomly initialized Qwen3 forwards, backward/optimizer steps, frozen-backbone gradients, FP16 backbone/FP32 adapter interoperability, non-reentrant gradient checkpointing, native causal/padding masks, grouped-query KV repetition, full-vs-cached decoding with nonzero interventions, identity initialization, adapter removal, prediction-preserving checkpoint round trips, split/oracle/parser behavior, per-bundle JSS, repeated-trial reliability, paired cluster CIs, partial coverage, integrity/resume guards, notebook payload round trip and process-group timeout cleanup.

A fresh independent whole-project review found two important bugs: training loss summaries read hash order, and a failed optimizer restore could leave an adapter attached to FULL. Both have fault-reproducing regression tests and fixes. A partial-run candidate seed coverage issue also has a regression test. These checks certify software behavior, not usefulness or novelty of the proposed learned operator.

Reproduce on a CUDA PyTorch environment with `python -m pip install -r requirements.txt` then `python -m pytest -q`. `--preset smoke` is intentionally TEST_ONLY, not a scientific benchmark. The Kaggle notebook runs these software checks before the pretrained experiment.

Additional integration: all 20 method configurations completed 160/160 random-Qwen TEST_ONLY requests and 160 offline score replays. The real Qwen3-4B-Instruct-2507 tokenizer/chat template was downloaded without model weights: 32 generated task/control examples passed token-count matching; longest teacher-forced sequence was 284 tokens, longest gold suffix 34 tokens (configured output cap 96). These are tokenizer/software checks, not pretrained inference results.
