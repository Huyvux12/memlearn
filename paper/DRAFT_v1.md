# The Later Source Reads the Earlier: Source Order, Rule Application, and Scope in Small Language Models

*Draft v1, 22 Oct 2026. Every number is produced by `paper/analysis.py` (layout `paper/layout_original.json`) from per-item receipts. Tables are in `paper/tables/`, figures in `paper/figures/`, and citation keys refer to `paper/refs.bib`. Pre-registrations are in `docs/`; the text marks post-hoc analyses explicitly.*

## Abstract

Small instruction-tuned language models are often handed a rule in context. Depending on the request, the rule must be *read* (described), *applied* (composed with a lookup), or *not applied* because the request is out of scope. We study these three behaviours on a controlled codebook-and-rule task, scored with teacher-forced candidate log-probabilities so that output format cannot affect any score. We test five frozen models from two architecture families (Qwen3-1.7B/4B/8B, Qwen2.5-3B, Phi-3.5-mini).

1. **Applying fails mainly by under-applying.** Models read a rule correctly when the first hop is supplied (78–95 %). When they must compose it with the lookup themselves, they mostly stop after the first hop (APPLY 12–66 %).
2. **Source order matters in the structured format.** Placing the rule before the codebook lowers rule reading in all five models (by 0.18–0.46). This asymmetry does **not** carry over to a prose version of the task.
3. **The later source must read the earlier one.** Blocking attention from rule tokens to codebook tokens removes rule reading in all five models, while a size-matched sham block does not (Δ −0.40 to −0.83). The dependency goes through the *shared value tokens*, not the keys, and it sits in early and middle layers. Blocking only **8 duplicate-token heads** removes application (Δ −0.51 to −0.57 against random heads). Restoring the clean rule-token states inside a mid-layer window brings rule reading back.
4. **Out-of-scope application depends on the model and the format.** Qwen3-4B never applies the rule out of scope. Phi-3.5-mini does so in 15–32 % of codebook-first cases, confirmed on fresh data. Smaller models leak more in prose. Describing the rule explicitly as a procedure *reduces* Phi's leakage, while a content-free sentence in the same position does not.

**Adapters.** We pre-registered a test of whether parameter-matched adapters that gate each value by its query–key pair before aggregation use the rule more selectively. **They do not:**
- they are no better than query-only, separable or LoRA adapters at composition or at repairing Phi's leakage;
- every adapter learns the trained source order but transfers poorly to the unseen one, even though training on both orders shows the capacity exists;
- under aggressive adaptation, value-branch adapters (separable or pairwise) cause less new out-of-scope application than LoRA, and the pairwise component adds nothing beyond the separable one (5 seeds).

## 1 Introduction

Models are often given rules in context: formatting conventions, policies, procedures. A useful model keeps three behaviours apart:
- saying what the rule is;
- applying it when the request authorizes it;
- leaving it alone when the request is out of scope.

Failure to apply and *scope leakage* (applying when not asked) look alike from the outside but have different causes.

We isolate the three behaviours with a synthetic task built for exact scoring:
- a **CODEBOOK** maps keys `x..` to values `z..`;
- a **RULE_MAP** P is a derangement of the values;
- an optional distractor rule Q competes with P;
- each mode uses its own input key.

Answers are scored by teacher-forcing the JSON prefix and comparing the log-probabilities of four candidates. This matters in practice. In free generation, small models often answer an apply request with the read schema (§4.4), so exact-match scoring would mostly measure format.

**Contributions.**
1. **A behavioural dissociation** between reading and applying, on five frozen models. Scope leakage depends on both the model and the format.
2. **A causal account of how the second source is used.** Necessity is shown by knockouts with sham controls. The mechanism is localized to shared value tokens, a layer band, and a handful of duplicate-token heads. Sufficiency is shown by restoration in a mid-layer window.
3. **A pre-registered negative result** for a pairwise query–key value-gating adapter, along with two findings about adaptation: it does not transfer across source order, and LoRA induces more scope leakage than value-branch adapters when trained aggressively.

## 2 Related work

- **In-context two-hop reasoning.** [guo2025twohop] show that pretrained LLMs given distractors fall back to random guessing among plausible conclusions. A few fine-tuning steps fix this, and they analyse the mechanism in a small transformer trained from scratch.‡ Latent multi-hop reasoning over *parametric* knowledge is studied by [press2023compositionality], [yang2024latent] and [biran2024hopping]; the last finds that the bridge entity resolves early and the second hop starts too late. [balesni2024twohop] report that models fail to compose two synthetic facts latently but succeed when one fact is natural.
- **Order sensitivity.** [chen2024premise] report drops of more than 30 % when premises are permuted away from proof order. [yu2025core] show that multi-hop performance depends on document order and that repeating the context helps. [ok2026lost] attribute a context-first versus options-first gap of more than 14 points to the causal mask. We add a causal knockout with sham controls, layer, token and head localization, and a sufficiency test. We also show that the order effect depends on the format.
- **Attention gating and adapters.** [qiu2025gated] gate the SDPA output with a head-specific, query-dependent sigmoid. [wang2025dva] makes values depend on the query; [zhao2026rca] rescales value contributions per pair at inference time. Pairwise, feature-wise gates on messages before aggregation already exist in [bresson2017rggc], [zhao2020san], [zhao2020pointtransformer] and [shen2018disan]. Our VECTOR_PAIR adapter belongs to this family. We do not claim the operator as new; we test it as a frozen-LLM adapter against LoRA [hu2021lora] and ReFT-style controls [wu2024reft]. Head-level attention steering is studied in [zhang2024pasta].

‡ Checked against the abstract; the claims that premise order is fixed and that pretrained models receive no attention knockout still need a check against the full text.

## 3 Task, models, and measurement

**Task.**
- Each bundle has 6 codebook entries, P, and Q.
- Modes: READ_DEC; READ_RULE (P of a given decoded value, one hop); APPLY_P and APPLY_Q (two hops); NONAPPLY (return the decoded value without applying P).
- Conditions: framing (neutral; default = P described as a standard procedure; filler = a content-free sentence in the same slot), distractor (none or Q), and source order (codebook first `bf` or rule first `rf`).
- Cue families: training/ID families and held-out families.
- **Prose variant:** badge numbers map to room names, and procedures are written as prose sentences ("room Amber moves to room Cedar"). It keeps the same candidate and JSON contract.

**Models.** Frozen fp16 Qwen3-4B-Instruct-2507, Qwen3-1.7B, Qwen3-8B, Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct.

**Probe and statistics.**
- The probe sums candidate log-probabilities from the longest token prefix that all candidates share.
- Metrics: choice accuracy, leakage (a NONAPPLY item answered with P(base)), stop-at-base, and the margin log p(gold) − log p(P(base)).
- Contrasts are paired by item. CIs are 95 % cluster bootstraps over bundles (seed × bundle for adapters).

## 4 Frozen models: reading survives, application fails, order matters (in one format)

### 4.1 Reading versus applying

*Table 1, `t1_frozen`; Figure 1.*

| Model | READ_RULE bf | READ_RULE rf | APPLY bf | APPLY rf | stop-at-base | Leak bf | Leak rf |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.95 | 0.73 | 0.64 | 0.36 | 0.38 | 0/192 | 0/192 |
| Qwen3-1.7B | 0.87 | 0.48 | 0.19 | 0.04 | 0.84 | 7/192 | 5/192 |
| Qwen2.5-3B | 0.78 | 0.32 | 0.12 | 0.08 | 0.82 | 0/192 | 3/192 |
| Phi-3.5-mini | 0.80 | 0.38 | 0.66 | 0.02 | 0.63 | 29/192 | 0/192 |
| Qwen3-8B (`t7_scale_frozen`) | 0.95 | 0.76 | 0.47 | 0.26 | 0.43 | 1/192 | 4/192 |

Given the first hop, every model reads the rule well. Composing it with the lookup is far worse, and the dominant error is returning the decoded value. In Qwen3-4B there are 156 bundle-conditions in which the rule is read correctly but not applied; the reverse happens in 25.

### 4.2 Source order

Putting the rule before the codebook lowers READ_RULE in all five models (Qwen3-8B: +0.18 [0.08, 0.27] for bf over rf).

**Proximity does not explain it.** Repeating the codebook right after the rule does not help: +0.08 [−0.02, 0.21] in Qwen3-4B.

**In the prose variant the asymmetry disappears** (pre-registered replication *not met*; `t6b_nl_frozen`). READ_RULE bf − rf is −0.09, −0.08, 0.00 and +0.03 across the four models, with every CI including 0. The read-over-apply gap holds in prose for Qwen3-4B (+0.40 [0.26, 0.55]) and Phi (+0.27 [0.11, 0.43]). It does not hold for Qwen2.5-3B (−0.17), whose prose rule reading is near floor. The order effect is therefore a property of how the structured sources interact, not a general law.

### 4.3 Scope

- **Qwen3-4B:** never applies the rule out of scope, in either format.
- **Phi-3.5-mini:** applies it in 15 % of codebook-first items. On fresh data, with a pre-registered threshold, the rate is 18.8 % (9/48). In the framing study it reaches 32 % for neutral framing.
- **Prose and small models:** prose raises leakage for small models (Qwen3-1.7B 29/192; Phi 17–18/192).
- **Framing (pre-registered control, `t9_framing`):** describing P as the standard procedure *lowers* Phi's leakage from 32 % to 18 %. A content-free sentence in the same slot leaves it at 31 % (default − filler = −0.14 [−0.22, −0.05]). The effect is therefore semantic, not positional.

### 4.4 Free generation agrees where it is measurable

*`t6_generation`, every 4th item.*

- Greedy generations that parse into one of the four candidates agree with the log-prob choice in 90–100 % of cases.
- Only 33–74 % parse at all. Small models mostly answer apply and non-apply requests with the *read* schema (`{"decoded_value","rule_result"}`), so free generation cannot measure the decision there.
- The order asymmetry also appears in generated READ_RULE answers (bf versus rf: 0.98 vs 0.81, 0.80 vs 0.54, 0.86 vs 0.17, 0.74 vs 0.36).

## 5 Mechanism: the later source must read the earlier one

We add −∞ to the attention logits from one source span to another during the prompt pass. The answer positions are untouched. A sham condition blocks an equal number of system-prompt tokens, and a guard checks that every targeted layer or head was hit.

### 5.1 Necessity

*`t2_knockout_read_rule`, `t7b_scale_knockout`; Figure 2.*

Blocking rule → codebook (bf) removes rule reading; the sham barely matters. Block − sham on READ_RULE:

| Model | Δ | CI95 |
|:---|---:|:---|
| Qwen3-4B | −0.71 | [−0.83, −0.58] |
| Qwen3-1.7B | −0.83 | [−0.94, −0.73] |
| Qwen2.5-3B | −0.54 | [−0.69, −0.40] |
| Phi-3.5-mini | −0.40 | [−0.56, −0.23] |
| Qwen3-8B | −0.71 | [−0.83, −0.58] |

- **Rule-first order.** Blocking codebook → rule also hurts (Qwen3-4B: −0.31 on READ_RULE), so in both orders rule content is accessed *through* the later source.
- **Phi's leakage runs through the same path.** In Phi, the same block lowers leakage from 31 % to 10 %.
- **Prose.** The knockout replicates in prose for Qwen3-4B (−0.52 [−0.67, −0.38]). The other models are near floor in prose rule reading, and their CIs include 0.

### 5.2 Which tokens and layers

*Qwen3-4B, `t3_e2_narrow`.* Blocking only the codebook *value* tokens does as much damage as blocking the whole codebook (APPLY −0.62, READ −0.54). Blocking only the *key* tokens does nothing (−0.02). Blocking layers 24–35 does nothing; blocking 0–11 or 12–23 removes application.

### 5.3 Heads

*`t5_e4`, pre-registered.* For each head we score duplicate-token attention: attention from a rule entry's left value to the same value in the codebook, minus the attention to the other values. Blocking only the **top-8 heads** (out of 768 candidate heads in Qwen3-4B and 640 in Phi) from codebook value tokens:

| Model | APPLY, top-8 vs random-8 | Δ [CI95] | READ_RULE, top-8 vs random-8 | Δ [CI95] |
|:---|:---|:---|:---|:---|
| Qwen3-4B | 0.04 vs 0.61 | −0.57 [−0.71, −0.43] | 0.56 vs 0.94 | −0.38 [−0.53, −0.23] |
| Phi-3.5-mini | 0.27 vs 0.78 | −0.51 [−0.67, −0.34] | 0.65 vs 0.88 | −0.23 [−0.35, −0.11] |

The heads lie in layers 1–19 (Qwen) and 5–19 (Phi). Application depends on them more than reading does.

### 5.4 Sufficiency

*`t5_e4`, Figure 4.* With the block in place, we write the clean rule-token states back after layer L.
- **Rule reading recovers** (Qwen3-4B 0.92–0.94 against 0.96 clean; Phi 0.81–0.85 against 0.90), but **only inside a mid-layer window**: L = 16–20 of 36 in Qwen, 12–18 of 32 in Phi.
- **Application recovers partly in Phi** (up to 0.62 against 0.79) and barely in Qwen (0.19 against 0.62).
- **Pre-registered prediction:** monotone recovery. **The shape was wrong.**
- **Our interpretation (post-hoc).** The rule tokens finish integrating the codebook by about the start of the window. Positions after the rule read those states only after the window, so restoring too late misses them. Application in Qwen additionally depends on downstream positions that read the rule earlier.

## 6 Adaptation: learns the order it sees; LoRA leaks most when pushed

*`t4_adaptation`, `t4b_leak_contrasts`; Figure 3.*

**Setup.** Parameter-matched adapters on the frozen backbone:
- QUERY: query-only gate on a low-rank value branch;
- SEPARABLE: φ(q)⊙ψ(k);
- VECTOR_PAIR: 1+tanh(Qq+Kk) applied to each value before aggregation, reusing the native attention weights;
- LoRA: on q/k/v/o, or on v alone.

Training runs 256 steps on codebook-first items. Results pool 5 seeds.

**No transfer across order (Qwen3-4B, 4 layers).**
- Trained on codebook-first only, adapters reach 0.90–0.96 in distribution but 0.43–0.63 on rule-first order (frozen: 0.33). LoRA varies most across seeds (0.33–0.75).
- Trained on both orders, they reach 0.70–0.94.
- In Phi, rule-first APPLY stays at ≤ 0.14.

**The architectural hypothesis (pre-registered, negative).**
- *Composition.* VECTOR_PAIR never beat the best control in any screen.
- *Repairing Phi's inherent leakage.* The frozen model leaks on 9 of 48 codebook-first items (18.8 %). After adaptation (5 seeds, n = 240 codebook-first NONAPPLY items per method) leakage falls to QUERY 11.7 %, SEPARABLE 10.4 %, VECTOR_PAIR 10.8 % and LoRA-qkvo 7.9 %. VECTOR_PAIR is not better than any control; none of its contrasts excludes 0 in its favour.

**Leakage induced by adaptation (Qwen3-4B, 5 seeds).**
- *Mild settings* (lr 5e-4 or 1e-3, codebook-first training): every method stays ≤ 2.3 %, with no differences.
- *Aggressive settings:* LoRA-qkvo leaks more than both value-branch adapters.
  - lr 2e-3: Separable − LoRA = −0.073 [−0.117, −0.033]; Vector pair − LoRA = −0.092 [−0.142, −0.046].
  - Both-order training: −0.098 and −0.104, both CIs excluding 0.
- *One layer at lr 2e-3:* SEPARABLE leaks less than VECTOR_PAIR (−0.033 [−0.060, −0.008]) and LoRA-v (−0.037).
- **Conclusion.** The protective effect comes from the *value-branch structure*, not from pairwise conditioning. The 3-seed impression that VECTOR_PAIR alone preserves scope did not survive 5 seeds.

## 7 Discussion and limitations

- **What generalizes.** The structural finding (the later source must read the earlier one, through shared value tokens and a few duplicate-token heads in early and middle layers) holds across 5 models and 2 families in the structured format. It also holds in prose for the strongest model. The *order asymmetry* is format-specific, and so is the *size* of the read/apply gap.
- **Scope.** Up to 8B parameters, one synthetic task family with a prose variant, English templates, and log-prob scoring. Free generation is unmeasurable for apply/non-apply in small models because of schema confusion.
- **Mechanism.** We provide necessity and restoration evidence, but no full path patching. The restoration window is interpreted post hoc.
- **Adapters.** Results depend on depth and learning rate, and we covered these only partly (1 or 4 layers; three learning rates).
- **Practical takeaways.**
  - Put the lookup source before the rule in structured prompts.
  - Do not expect fine-tuning on one layout to transfer to another.
  - Prefer value-branch adapters over LoRA when scope must be preserved under aggressive adaptation.

## 8 Reproducibility

- **Code.** The released package (`README.md`) contains the task generator, probes, knockouts, restoration and head analysis, adapters, a parallel single-GPU job runner, `tools/make_job_specs.py` (a 58-stage spec that reproduces every experiment), and `paper/analysis.py` with layouts for the original and reproduced runs.
- **Integrity.** Per-item receipts are checksummed, and every run records the model revision and the source hash.
- **Compute.** About 2.5 GPU-hours on a T4 and about 22 Colab compute units on an RTX PRO 6000.

## Remaining before submission

1. Full path patching from the duplicate-token heads to the answer position, to turn the window account into a circuit.
2. A non-Qwen/Phi family (Llama-3.x needs a license token) and a second prose template.
3. Full-text checks of the claims marked ‡, plus the venues listed as unconfirmed in `refs.bib`.
