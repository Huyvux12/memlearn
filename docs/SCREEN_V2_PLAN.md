# Phase C screen on benchmark v2 — pre-registered plan (2026-10-22)

Code: `memlearn_attention/screen_v2.py`. Background: `runs/probe_v2_kaggle/ANALYSIS.md`. This plan is fixed before any adapter is trained on v2. It is a **screen** (1 seed): it decides which methods go on to a 3-seed confirmatory run. It does not decide the paper claim.

## Setup

- **Model:** frozen `Qwen/Qwen3-4B-Instruct-2507`, revision recorded per run.
- **Adapter placement:** one adapter in the middle layer, with ranks matched to the parameter count of VECTOR_PAIR.
- **Training:**
  - Items come only from `book_first` and cover every mode (READ_DEC, READ_RULE, APPLY_P, APPLY_Q, NONAPPLY).
  - Both framings and both distractor settings are used.
  - Loss is next-token CE on the teacher-forced JSON continuation plus EOS.
  - Steps and learning rate are identical for all methods.
- **Evaluation:**
  - Neutral framing, distractor ∈ {none, q} × source order ∈ {book_first, **rule_first**}.
  - Modes: READ_DEC, READ_RULE, APPLY_P, NONAPPLY.
  - Splits test_id and test_ood. Scoring uses log-probabilities over 4 candidates (`probe_v2`).
- **Held-out axis:** `rule_first` never appears in training. On the frozen model it is the cell where composition fails (APPLY_P 0.27).

## Methods and run groups

| Group | Methods | Question |
|:---|:---|:---|
| A | FULL, QUERY, SEPARABLE, VECTOR_PAIR, LORA_V | Candidate against the nearest controls, plus a standard PEFT |
| B | KEY, SCALAR_PAIR, POST_GATE, LORA_O, LORA_QK | Remaining operator classes |

Both groups use the same data seed, so their items are identical. The comparison across groups is paired by item id.

## Primary outcome

**APPLY_P choice accuracy on `rule_first`**, pooled over both test splits and both distractor settings.

Primary contrasts, paired by item, with a bootstrap over bundles:
1. VECTOR_PAIR − QUERY
2. VECTOR_PAIR − SEPARABLE
3. VECTOR_PAIR − best LoRA

## Preservation constraints (checked for every method)

| Measure | Requirement |
|:---|:---|
| NONAPPLY accuracy | ≥ FULL − 0.02 (FULL is 1.00 on the probe) |
| NONAPPLY margin log p(base) − log p(P) | Report the median. A drop of more than half relative to FULL is flagged as **adaptation-induced leakage risk**, even if no argmax flips |
| READ_DEC and READ_RULE accuracy | ≥ FULL − 0.02 |

## Secondary outcomes

- APPLY_P on `book_first`: does the method learn the task at all?
- The gap between `book_first` and `rule_first`: does the method remove the order effect?
- JSS-v2 per condition.
- Test_ood against test_id: transfer to unseen cue families.
- Cost: trainable parameters, training seconds per step, probe seconds per item.

## Decision rule for the screen

| Observation | Decision |
|:---|:---|
| No trained method improves rule_first APPLY_P by ≥ 0.15 over FULL while meeting the constraints | Frozen-adapter route does not fix composition at this budget. Revisit the steps/layer budget once, then report as negative |
| Some methods improve it, and VECTOR_PAIR ≥ the best control + 0.10 | Go to 3-seed confirmation with VECTOR_PAIR and the two strongest controls |
| A simpler method (QUERY, SEPARABLE, LoRA) ties or wins | Go to confirmation with the simplest winner. The paper story becomes "minimal sufficient conditioning" |
| Any method meets the gain but breaks a constraint | Record it as a trade-off. That method is not eligible as the winner |

## Budget (estimate; measured in the first run)

- Measured on T4 by earlier runs: training about 0.37 s/step, probe 0.47 s/item before the candidates were batched.
- Planned per group: 256 training steps per method and 12 + 12 test bundles, giving 384 probe items per method.
- Estimate: about 4 minutes per trained method, about 20 minutes per group including setup. Every notebook has a self-stop and an always-export ZIP.

## Revisit run (added 2026-10-22, before the run; the one permitted revisit)

The trigger is the group-A outcome: no method met a gain of at least 0.15 while satisfying the constraints (`runs/screen_v2_A_kaggle/ANALYSIS.md`).

**Changes:**
- Adapters in 4 layers, {6, 12, 18, 24}, instead of the single middle layer. The mechanism run showed that cross-source integration happens in the prefill of the later source.
- Learning rate 5e-4 instead of 2e-3. Group A reached a training loss near 0 together with leakage inside the training distribution.
- `LORA_QKVO` (LoRA on q, k, v and o, parameter-matched) replaces LORA_V, following the setting of Guo et al. 2025.

**Unchanged:** data, items, steps (256), the probe, the constraints and the decision rule.

**Methods, in run order:** FULL, VECTOR_PAIR, QUERY, SEPARABLE, LORA_QKVO.

**Outcome:** if this run also fails the rule, the adapter-fixes-composition claim is reported as negative at this budget. The paper focus then moves to (a) the mechanism result and (b) adaptation-induced scope leakage by operator class, with 3 seeds.
