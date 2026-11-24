# Benchmark v2 and format-free probe — specification (2026-10-22)

## Why

Pilots v1 and v1.1 (runs/*/ANALYSIS.md) showed two problems:
- The frozen Qwen3-4B never applied the rule out of scope (0/36 NONAPPLY).
- Its errors were JSON-format errors, so exact-JSON scoring measured format compliance.

After adaptation the task saturated. v2 asks a narrower first question, before any adapter is trained: **does the frozen model show the target failure when the rule is under real pressure to be applied?** It measures the decision with log-probabilities, not free generation.

## Task (memlearn_attention/data_v2.py)

### Bundle contents

Each bundle has:
- a codebook of 6 keys (x00–x99) mapped to 6 values (z00–z99);
- a rule **P**, a derangement of the values ($P(v) \ne v$);
- a distractor rule **Q**, also a derangement, with $Q(v) \ne P(v)$ for every value.

### Keys per mode

Every mode uses its **own input key**, so no mode can copy another mode's answer. The modes are:
- READ_DEC: decoded value;
- READ_RULE: hypothetical $P$(decoded);
- APPLY_P;
- APPLY_Q (only when Q is shown);
- NONAPPLY.

### Conditions

The same bundle is rendered under every condition, so comparisons are paired.

| Factor | Levels | Purpose |
|:---|:---|:---|
| framing | `neutral` (RULE_MAP only), `default` (P is described as the standard procedure applied to every decoded value) | Pressure to apply P |
| distractor | `none`, `q` (Q shown as a second procedure) | Selective use among competing rules |
| source_order | `book_first`, `rule_first` | Position control |

The request wording comes from the v1 cue families: train/ID cues and the held-out OOD cues.

## Probe (memlearn_attention/probe_v2.py)

### Scoring a request

The probe teacher-forces the JSON prefix, for example `{"answer":"`. Each candidate is scored as the summed log-probability of the continuation `cand"}`, measured from the longest token prefix that all candidates share. That makes tokenization merges at the boundary harmless.

Candidates per mode:

| Mode | Candidates |
|:---|:---|
| APPLY_P, APPLY_Q, NONAPPLY | base, P(base), Q(base), and one other codebook value. Q(base) is still scored as a candidate when Q is hidden; it is then a plain codebook value |
| READ_DEC | base, the input key itself (the v1 failure), P(base), and one other value |
| READ_RULE | given the correct decoded value: P(base), base, Q(base), and one other value |

### Metrics per (method, condition, mode)

- **choice accuracy:** the argmax candidate is the gold answer.
- **leakage:** a NONAPPLY item whose argmax is P(base).
- **cross-rule confusion:** an APPLY_P item whose argmax is Q(base).
- **margins:** log p(gold) − log p(P(base)) on NONAPPLY.
- **joint selective score (JSS-v2):** the bundle is correct on READ_DEC ∧ APPLY_P ∧ NONAPPLY within the same condition.

### Methods in the first probe

FULL and SCOPE_PROMPT (frozen model). Adapters come later, through the same probe.

## Decision rule (fixed before the GPU run)

The target failure **exists** if the frozen FULL model has NONAPPLY leakage ≥ 10 % in at least one condition cell (n ≥ 32 bundles per cell), with APPLY_P choice accuracy ≥ 70 % in that cell.

- **If it exists:** use those cells as the primary benchmark for phase C. Train on `neutral`/`none` cells only, and keep the other cells as held-out pressure.
- **If not:** the frozen model handles scope by itself in this family of tasks. The research question then moves to adaptation-induced leakage, which pilot v1.1 hinted at for LORA_V.

## Not in this version

- The feature-binding task, which tests the pre/post-aggregation hypothesis directly. It is planned for v2.1, after the probe.
- Training adapters on v2.
