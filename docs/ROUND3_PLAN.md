# Round 3 — pre-registered predictions (written 22 Oct 2026, before the run)

Code: v2.0.0. One Colab job runs all stages in parallel (`deploy/colab_round3/job.json`). Scoring uses the 4-candidate log-prob probe throughout. CIs are paired cluster bootstraps.

## E4 — sufficiency and head localization (`mech_e4`; Qwen3-4B and Phi-3.5-mini)

**Setup.** Codebook first, neutral framing, no distractor, APPLY_P and READ_RULE, 24 + 24 bundles (`data_seed = 17`).

**(a) Restoration.** We block rule ↛ codebook in all layers and put back the clean rule-token states after layer L.
- *Prediction:* accuracy recovers to within 0.10 of the clean run once L is past the band in which the binding forms, and does not recover for small L.
- *For Qwen3-4B:* recovery by L ≈ 24 (E2 showed that blocking layers 24–35 has no effect).
- *Readout:* the smallest L at which accuracy reaches clean − 0.10, reported for APPLY and READ separately.

**(c) Head knockout.** We block the top-k duplicate-token heads (k ∈ {8, 32}, within the first 2/3 of the layers) from rule tokens to codebook value tokens, and compare with the mean of 3 random-k draws.
- *Prediction:* top-k lowers APPLY or READ_RULE by ≥ 0.20 more than random-k, for at least one k.
- *If not met:* we report that the binding is distributed across heads and that it is not localized at k ≤ 32.

## E5 — external validity

**(a) Natural-language variant** (`style = 'nl'`: badge → room directory, prose procedures). We run probe + M0–M5 on Qwen3-4B, Qwen3-1.7B, Qwen2.5-3B and Phi-3.5-mini (`data_seed = 41`).

Predictions, each counted as replicated if the sign matches and the CI excludes 0 in at least 3 of the 4 models:
1. READ_RULE (bf) > APPLY_P (bf).
2. READ_RULE bf > rf.
3. M1 − M2 < 0 on READ_RULE.

**(b) Free generation.** On every 4th probe item for the 4 models (codes style), we generate greedily and parse the answer.
- *Agreement* = the generated value equals the log-prob choice, excluding invalid outputs.
- *Prediction:* agreement ≥ 0.80.
- *Always reported:* the invalid rate, and the conclusions (APPLY < READ, the order effect, leakage) recomputed on the generated answers.

## E6 — scale (Qwen3-8B)

Probe + M0–M5 with codes, `data_seed = 17`.

**Prediction:** the order asymmetry (READ_RULE bf > rf) and M1 − M2 < 0 on READ_RULE both hold. We make no prediction on APPLY or leakage levels.

## E7 — leakage statistics (5 seeds) and lr sweep

**Setup.**
- Seeds 7 and 11 are added to: E3 bf / both / bfhi (Qwen3-4B, 4 layers), Q1L (Qwen3-4B, 1 layer, lr 2e-3), and Phi P (`data_seed = 23`, 4 layers, lr 5e-4).
- New run: lr 1e-3 (Qwen3-4B, 4 layers, train bf) with seeds 17, 29, 43, 7, 11.

**Reporting.** Leakage is pooled over 5 seeds with cluster CIs. The rankings already reported (3 seeds) are re-tested with 5 seeds, and the result is reported whatever it is.

## E8 — Phi framing control

**Setup.** Phi-3.5-mini, codes, `data_seed = 31`. Framing ∈ {neutral, default, filler} × distractor ∈ {none, q}, codebook first, all modes. Filler is a content-free sentence of similar length placed in the same slot as the default framing.

**Primary outcome.** NONAPPLY leakage per framing.

**Predictions:**
- If leak(filler) ≈ leak(default) < leak(neutral), with the CI of default − filler including 0, the framing effect is positional or due to separation.
- If leak(filler) ≈ leak(neutral) > leak(default), it is semantic.

Either outcome is reported.
