# Round 4 — pre-registered predictions (written 22 Oct 2026, before the run)

**Goal:** raise the paper from workshop level to Findings or main track by closing the four gaps reviewers will raise:
- G1: why prose loses the order asymmetry;
- G2: a circuit rather than necessity alone;
- G3: model families beyond Qwen and Phi;
- G4: prose results resting on a single template.

**Setup.** Code v2.1.0; one Colab job (`deploy/colab_round4/job.json`). Scoring uses the 4-candidate log-prob probe; CIs are paired cluster bootstraps.

## G2 — circuit: binding heads → reader heads (`mech_e4`, codes, codebook first)

**Models:** Qwen3-4B, Qwen3-8B, Phi-3.5-mini.

**New measurements:**
- **Binding-head sufficiency (`only_top{k}`).** Rule tokens are blocked from the codebook in *every* head except the top-k binding heads.
- **Reader heads.** Score each head by its attention from the last prompt position (which predicts the answer) to the tokens of the rule entry for the queried value, minus its mean attention to the other entries. Then block the top-k reader heads, and a random-k set, from the rule tokens at the answer positions.
- **Path.** The mean reader-head attention to the correct entry in three runs: clean, with the top-k binding heads blocked, and with random-k binding heads blocked.

**Predictions** (k = 8 for necessity, k = 32 for sufficiency):
- **P2a.** `only_top32` keeps READ_RULE within 0.15 of clean, while the all-head block is at the level measured in E2. *If this fails:* binding is distributed beyond 32 heads.
- **P2b.** Blocking the top-8 reader heads lowers APPLY and READ_RULE by ≥ 0.20 more than random-8.
- **P2c.** Blocking the binding heads lowers the readers' attention to the correct entry more than blocking random heads does (CI excludes 0). This links the two stages causally.

## G1 — why prose differs (`mech_e4` with `style = 'nl'` on Qwen3-4B and Qwen3-8B, both orders; codes rule-first on Qwen3-4B for comparison)

**Hypothesis H-prose.** In prose, each entry repeats an explicit relational frame ("badge K is assigned to room V"; "room V moves to room W"). The shared value token is then reachable by attention in *either* direction, so the later source integrates the earlier one equally well in both orders. In the JSON format, the integration is effective only when the rule comes second.

**Predictions:**
- **P1a.** In prose, restoration recovers READ_RULE in *both* orders. In codes, rule-first restoration (blocking codebook ↛ rule) recovers less than codebook-first.
- **P1b.** The binding heads found in prose overlap with the binding heads found in codes, measured by the Jaccard index of the top-32 sets, above the overlap of random sets.

**Readout.** Either outcome is reported. If P1a fails, H-prose is not supported, and the format effect stays descriptive.

## G3 — new families and scale (codes probe + knockout M0–M5; prose probe)

**Models:** OLMo-2-1124-7B-Instruct and Granite-3.3-8B-Instruct (new families, not gated); Qwen3-14B (scale).

**Predictions:** READ_RULE bf > APPLY bf, READ_RULE bf > rf, and M1 − M2 < 0 on READ_RULE, each with a CI excluding 0. These are counted as replicated if they hold in ≥ 2 of the 3 models.

## G4 — two more prose templates (`nl_gate`: flights → gates; `nl_shelf`: items → shelves)

**Models:** Qwen3-4B, Qwen3-8B, Phi-3.5-mini, OLMo-2-7B, Granite-3.3-8B. Each runs the probe and knockout M0–M5, with `data_seed = 53`.

**Predictions:**
- (a) READ_RULE bf > APPLY bf in ≥ 3 of 5 models, for each template.
- (b) The order asymmetry stays absent (|READ_RULE bf − rf| < 0.10, or a CI including 0) in ≥ 3 of 5 models. This tests whether the prose finding generalizes across templates.
- (c) The knockout M1 − M2 < 0 holds in every model whose prose READ_RULE is ≥ 0.6.
