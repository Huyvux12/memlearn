# Mechanism test of the source-order effect — pre-registered predictions (2026-10-22)

Code: `memlearn_attention/mech_v2.py`. Background: on the frozen model, APPLY_P is 0.60 when CODEBOOK comes first and 0.27 when RULE_MAP comes first (`runs/probe_v2_kaggle`).

## Setup

- **Model:** frozen Qwen3-4B. Framing is neutral and there is no distractor.
- **Data:** 24 + 24 test bundles. Modes APPLY_P, NONAPPLY and READ_RULE, scored with the 4-candidate log-prob probe.
- **Intervention:** additive −inf from the blocked query span to the blocked key span, in every layer and every head. It applies only to the prompt forward pass, so the answer positions still see everything.
- **Sham:** blocks the same number of key tokens, taken from the middle of the system contract.
- **Guard:** the run aborts if the block was not applied in every layer.

| Variant | Order | Block | What it tests |
|:---|:---|:---|:---|
| M0 | book_first | none | Reference |
| M1 | book_first | rule tokens ↛ codebook tokens | Do rule tokens pre-compose with the codebook? |
| M2 | book_first | rule tokens ↛ equal-size sham | Generic disruption control for M1 |
| M3 | rule_first | none | Reference (failure cell) |
| M4 | rule_first | codebook tokens ↛ rule tokens | Do codebook tokens pre-compose with the rule? |
| M5 | rule_first | codebook tokens ↛ equal-size sham | Control for M4 |
| M6 | rule, codebook, codebook | none | Proximity of the codebook to the answer, rather than composition |

## Hypotheses and predictions on APPLY_P accuracy

- **H1, pre-composition at the later source in book_first order.** Rule tokens read the codebook and store the composed mapping. Prediction: M1 ≪ M0 (drop ≥ 0.15, down toward M3), with M2 ≈ M0 (|Δ| < 0.05).
- **H2, composition at the answer position only.** Prediction: M1 ≈ M2 ≈ M0, and the order effect needs another explanation.
- **H3, codebook tokens in rule_first order already use the rule.** Prediction: M4 < M3 by ≥ 0.10, with M5 ≈ M3. If M4 ≈ M3, the codebook tokens do not exploit the earlier rule. That fits the asymmetry: the composed value is useful only where the *query key* lives (the codebook), and in rule_first order the codebook would have to pre-compose forward without knowing which entry will be asked.
- **H4, recency/proximity.** Prediction: M6 > M3 by ≥ 0.15.

## Safety checks

- NONAPPLY must remain correct in every variant (rule leakage stays at 0). Any leakage under a block is reported.
- READ_RULE is reported as a check that one-hop lookup survives the block. A large drop means the block broke more than composition.

Bootstrap over bundles, paired by bundle. A single run of the frozen model; the result is deterministic up to GPU numerics.
