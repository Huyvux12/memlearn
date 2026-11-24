# MEMLEARN: reading vs. applying in-context rules, source order, and attention adapters

Code, local experiment plans, and analysis for *"Binding Makes Rules Addressable: Key-Mediated Retrieval of In-Context Rules"* ([manuscript v5](paper/DRAFT_v5.md), [PDF](output/pdf/MEMLEARN_manuscript_v5.pdf)). The earlier v4 manuscript and release archives are retained for provenance.

Every result in the paper comes from a synthetic codebook-and-rule task, scored by **teacher-forced candidate log-probabilities**, so answer preference is measured separately from free-generation schema failures. Thirteen open models from seven families (counting Qwen2.5 and Qwen3 together) run frozen, optionally with parameter-matched adapters attached inside attention.

**Current paper revision (17 November 2026).** V5 uses the existing experiments only: no new pretrained inference, training or GPU jobs. CPU reanalysis repairs condition matching, rejects duplicate paired keys, and adds three consolidated tables (36 total). The primary positive result is key-mediated rule retrieval; full application recovery is less consistent, and the pairwise-gating hypothesis is not supported consistently. See [revision notes](paper/REVISION_v5.md).

To rebuild the paper from the raw receipts already present locally:

```bash
pip install -r paper/requirements.txt
python -X utf8 -m pytest -q tests/test_paper_analysis.py
python -X utf8 paper/analysis.py --layout paper/layout_original.json
python -X utf8 paper/build_manuscript.py
python -X utf8 paper/verify_manuscript.py
```

These paper commands do not load model weights or start GPU experiments.

| | |
|:---|:---|
| Language | Python ≥ 3.10 |
| Core dependencies | PyTorch ≥ 2.6, **transformers == 4.57.1** (pinned: the attention hooks rely on this version's eager-attention API) |
| GPU | Needed for real runs; ~10 GB VRAM per 4B-scale stage, ~20 GB for 8B models and ~30 GB for Qwen3-14B. Everything also runs on CPU with tiny random models (smoke presets). |
| Secrets | Only for Gemma-2, which is gated on Hugging Face: accept its license there and `export HF_TOKEN=...` before running its stages. Llama-3 is loaded from the ungated `unsloth` mirrors (same weights; the Llama 3.1/3.2 Community License applies). Every other model downloads without a token. |
| License | MIT (see `LICENSE`) |

## 1. Quickstart (CPU, ~5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build of torch >= 2.6
pip install -r requirements.txt
python -m pytest -q          # 2 process-supervisor tests use os.killpg and only pass on Linux/macOS

# Tiny random-model smoke runs of every experiment module (CPU):
python -m memlearn_attention.probe_v2  --preset probe_smoke  --out runs/smoke_probe
python -m memlearn_attention.mech_v2   --preset mech_smoke   --out runs/smoke_mech
python -m memlearn_attention.screen_v2 --preset screen_smoke --out runs/smoke_screen
python -m memlearn_attention.mech_e4   --preset e4_smoke     --out runs/smoke_e4
python -m memlearn_attention.mech_path --preset path_smoke   --out runs/smoke_path
```

Every run writes per-item **receipts** (`<out>/<kind>/*.json`, each checksummed) together with `config.json`, `manifest.json` (source hash, model revision, GPU) and `run_status.json`. A rerun with the same `--out` resumes and never duplicates a receipt. Changing the configuration requires a new output directory.

## 2. Running one experiment on a GPU

Each module has a preset and accepts a JSON file of overrides:

```bash
echo '{"model": "Qwen/Qwen3-1.7B", "allow_any_arch": true}' > my.json
python -m memlearn_attention.probe_v2 --preset probe_v2 --config my.json --out runs/probe_qwen3_1p7b
```

| Module | What it does | Key options |
|:---|:---|:---|
| `memlearn_attention.data_v2` | Task generator: codebook (6 entries), rule P (a derangement), distractor rule Q, a distinct input key per mode. Modes: READ_DEC, READ_RULE, APPLY_P, APPLY_Q, NONAPPLY. | `style`: `codes`, or prose `nl`, `nl_gate`, `nl_shelf`; framing `neutral`, `default` or `filler`; order `book_first` or `rule_first` |
| `memlearn_attention.probe_v2` | Frozen-model probe: 4-candidate log-probabilities after a teacher-forced JSON prefix | `model`, `data_seed`, `style`, `conditions`, `modes`, `generate_every` (free-generation check) |
| `memlearn_attention.mech_v2` | Attention knockout: −∞ from one source span to another during the prompt pass, with a size-matched sham | `variants` (M0–M6, N3–N7), `style`; presets `mech_v2`, `mech_e2` |
| `memlearn_attention.mech_e4` | Restoration (put clean later-source states back after layer L); binding-head (duplicate-token) and reader-head scores; necessity (top-k vs random-k), sufficiency (only top-k may bind) and binding → reader path | `style`, `order`, `ks`, `suff_ks`, `read_ks`, `random_draws`, `restore_every`, `head_depth` |
| `memlearn_attention.mech_path` | Round 5. Sender side: add back, after layer L*, the clean-minus-blocked direct contributions of chosen heads/MLPs at the rule tokens. Receiver side: give the reader heads' key/value groups their clean K and/or V at the rule tokens. Route: whole-codebook vs value-token vs top-k binding-head blocks, with reader attention. Heads (round 6): clean keys for single query heads only, separating reader heads from their key/value-group mates. Heads are ranked on held-out bundles | `parts`, `path_layer`, `dec_ks`, `dec_rand_ks`, `bind_k`, `read_k`, `head_ks`, `head_seed`, `style` |
| `memlearn_attention.screen_v2` | Trains parameter-matched adapters on the frozen model (codebook-first items), then probes both orders | `methods`, `layers`, `lr`, `seed`, `train_conditions`, `train_steps` |
| `memlearn_attention.attention` | Adapter operators and `attach()` for Qwen3 and Phi3 (the probes and knockouts work with any HF decoder exposing `eager_attention_forward`, e.g. Qwen2, OLMo-2, Granite): QUERY, KEY, SEPARABLE, SCALAR_PAIR, VECTOR_PAIR, POST_GATE, LoRA (V / O / QK / QKVO), … | — |

Older modules (`data.py`, `runner.py`, `metrics.py`) are the v1 pipeline used by the pilots. They are kept for provenance, and `tools/package.py` still builds their notebooks.

## 3. Reproducing every experiment in the paper

```bash
python tools/make_job_specs.py     # writes deploy/reproduce_all/job.json (131 stages) and paper/layout_reproduce.json
```

The job spec lists every stage (module, preset, overrides). `tools/colab_job.py` runs the stages **in parallel on one GPU**: each stage is its own process, models are downloaded once into the shared Hugging Face cache, the test suite runs once, and every stage self-stops at its time budget. Stages that fail (typically a CUDA out-of-memory error while several large models share the GPU) are retried once, sequentially, on the otherwise idle GPU, resuming from their receipts.

**On any Linux machine with a GPU:**

```bash
python tools/colab_job.py deploy/reproduce_all/job.json --root . --src ""   # results_reproduce_all.zip is written next to --root
```

Set `"parallel"` in the spec to fit your VRAM: about `VRAM_GB / 12` for the 4B-scale stages. Optionally give each stage a `"vram_gb"` estimate and the spec a `"vram_budget_gb"`: a stage then starts only when the running stages leave room (the round-5 spec does this).

**On Google Colab via the `colab` CLI** (`colab new`, `upload`, `exec`, `download`, `stop`):

```bash
python -c "import runpy,zipfile; f=runpy.run_path('tools/package.py')['_plain_files'](); z=zipfile.ZipFile('memlearn_src.zip','w'); [z.writestr(k,v) for k,v in f.items()]"
colab new -s repro --gpu G4
colab upload -s repro memlearn_src.zip content/memlearn_src.zip
colab upload -s repro tools/colab_job.py content/colab_job.py
colab upload -s repro deploy/reproduce_all/job.json content/job.json
echo 'import subprocess; subprocess.Popen("cd /content && nohup python -u colab_job.py job.json > job.log 2>&1 &", shell=True)' | colab exec -s repro
# poll /content/job.log until "JOB DONE", then:
colab download -s repro content/results_reproduce_all.zip runs/reproduce_all/results_reproduce_all.zip
colab stop -s repro
```

**Rebuilding every table and figure:**

```bash
python paper/analysis.py --layout paper/layout_reproduce.json --out my_results   # -> my_results/tables, my_results/figures
```

`--runs` (default `runs/`) is the folder the layout paths are relative to. Without `--out`, the outputs overwrite `paper/tables` and `paper/figures`.

`paper/layout_original.json` maps the original runs instead. Missing sources are listed in `<out>/tables/NOTES.md`; the script never invents them.

**Raw receipts of the original runs (no GPU needed).** The authors distribute them separately as `memlearn_raw_runs_v2.3.0.bin`, a zip archive of 219 MB, 21,272 entries, SHA256 `55fd2767737cf8143f529ce24a4f0e3c409dcc64615e84fe4e964be6fa44c748`. It currently lives in a private dataset; ask the authors for access. To use it:

```bash
sha256sum memlearn_raw_runs_v2.3.0.bin            # must match the value above
python -c "import zipfile; zipfile.ZipFile('memlearn_raw_runs_v2.3.0.bin').extractall('.')"   # creates runs/...
python paper/analysis.py --layout paper/layout_original.json --out check
```

The original v4 analysis produced 33 tables. In the 17 November audit, all Markdown tables reproduced; four CSVs differed only in floating-point rounding (maximum 2.22e-16). V5 repairs the framing match in prediction contrasts, adds receiver diagnostic contrasts and three summary tables. Its outputs intentionally differ from the frozen v2.3.0 release summaries; rebuilding the old release is a separate historical check.

**Kaggle.** `python tools/package.py --plain --preset e1 --overrides '{"model": "Qwen/Qwen3-1.7B"}' --out deploy/k` builds a readable, self-contained notebook (every source file in a `%%writefile` cell, with a SHA256 check) that self-stops at `--seconds`.

### Experiments, settings, and cost

Approximate wall time per stage; parallel stages share one GPU.

| ID | Question | Stages in `reproduce_all` | Approx. time per stage (RTX PRO 6000 / T4) |
|:---|:---|:---|:---|
| B/E1 | Frozen behaviour: read vs apply vs non-apply, source order (4 models) | `probe_*`, `mech_*` | probe 4–6 / ~15 min; knockout 2 / ~5 min |
| E2 | Narrow knockouts: layer bands, key vs value tokens (Qwen3-4B) | `e2_qwen3_4b` | 2 / ~6 min |
| E3/E7 | Adapters × source order × learning rate (Qwen3-4B 4 layers and 1 layer; Phi-3.5), 5 seeds | `full`, `bf_s*`, `both_s*`, `bfhi_s*`, `lr1e3_s*`, `q1l_s*`, `phi_*` | ~10 min (3–4 methods) / ~15 min |
| E4 | Sufficiency (restoration) and head localization | `e4_*` | see `paper/tables/t5_e4.md` |
| E5 | Natural-language variant; free-generation agreement | `nl_*`, `gen_*` | probe-like |
| E6 | Scale (Qwen3-8B) | `q8b_*` | probe-like, ~20 GB VRAM |
| E8 | Phi framing control (neutral / default / filler) | `filler_phi35_mini` | probe-like |
| R4-G2 | Circuit: binding-head sufficiency, reader heads, binding → reader path | `e4c_*` | ~13 min (3B–4B) |
| R4-G1 | Prose vs JSON mechanism, both orders | `e4_nl_*`, `e4_codes_rf_*` | ~13 min |
| R4-G3 | New families and scale: OLMo-2-7B, Granite-3.3-8B, Qwen3-14B | `probe_*`, `mech_*`, `nl_probe_*` | probe-like; 14B ~30 GB |
| R4-G4 | Two more prose templates (flights → gates, items → shelves) | `nl_gate_*`, `nl_shelf_*` | probe-like |
| R5-G5/G6/G7 | Sender decomposition (heads vs MLPs), reader-head K/V restoration, the prose route (fresh bundles, heads ranked on held-out bundles) | `path_*` | ~15–30 min |
| R5-G8 | Gemma-2-9B/2B (gated, bfloat16) and Mistral-7B: probe, knockout, prose, circuit | `probe_*`, `mech_*`, `nl_probe_*`, `e4c_*` | probe-like |
| R6-G9/G10 | Keys per query head; receiver replication on fresh bundles | `heads_*` | ~10–15 min |
| R6-G11 | Llama-3.1-8B (full path analysis, circuit) and Llama-3.2-3B | `*_llama31_8b`, `*_llama32_3b` | probe-like |

## 4. Determinism and caveats

- **Data.** Bundles are a deterministic function of `(data_seed, split, index, family, style)`, so every method and model sees identical items, and contrasts are paired by item.
- **Adapters.** Training is seeded (`seed`), but GPU numerics differ across hardware. Frozen-probe results matched to the reported precision between a T4 and an RTX PRO 6000. Single-seed **leakage counts** of a few items did not always match; report multi-seed pools.
- **Pre-registration.** Pre-registrations are in `docs/` (`BENCHMARK_V2_SPEC.md`, `SCREEN_V2_PLAN.md`, `MECH_V2_PLAN.md`, `PAPER_A_PLAN.md`, `PHI_PLAN.md`, `ROUND3_PLAN.md`, `ROUND4_PLAN.md`, `ROUND5_PLAN.md`, `ROUND6_PLAN.md`). The paper marks post-hoc analyses as such.
- **Statistics.** CIs are paired cluster bootstraps over bundles (and seed × bundle).

## 5. Layout

```
memlearn_attention/   task generator, probes, knockouts, adapters, training (v2) + v1 pipeline
tools/                colab_job.py (parallel runner), make_job_specs.py, package.py (notebook builder), notebook_runtime.py
tests/                pytest suite (CPU, tiny random models)
docs/                 specs and pre-registrations
paper/                analysis.py, layouts, refs.bib, draft
deploy/               job specs (reproduce_all, rounds 3-6)
results_summary/      tables, figures and result-archive hashes of the original runs (release package only)
```

## 6. Citation

See `paper/DRAFT_v5.md` and `paper/refs.bib`. Authorship and venue metadata should be supplied by the project authors before submission; the current PDF omits an author block.
