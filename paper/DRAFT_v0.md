# The Later Source Reads the Earlier: Source Order, Rule Application, and Scope in Small Language Models

*Draft v0, 22 Oct 2026. Every number below is produced by `paper/analysis.py` from per-item receipts in `runs/`. Table and figure files are in `paper/tables/` and `paper/figures/`. Citations marked † still need to be checked against the full papers; the literature screen in `MEMLEARN_NOVELTY_CHECK_REPORT_2026-10-22.md` relied partly on search snippets.*

## Abstract

We study how small instruction-tuned language models use a rule that is stated in context. The rule may need to be *read* (described), *applied* (composed with a lookup), or *not applied* (ignored because the request is out of scope). We use a controlled codebook-and-rule task, scored by candidate log-probabilities so that output format plays no role, and test four frozen models from two architecture families. Four findings stand out.

1. **Applying a rule fails mostly through under-application.** The frozen models apply the rule correctly in only 12–66 % of cases when the codebook precedes the rule, and they usually stop at the first hop. Yet they read the same rule correctly 78–95 % of the time once the first hop is supplied.
2. **Rule use depends strongly on source order.** Placing the rule before the codebook lowers rule reading by 0.22–0.46 and rule application to 0.02–0.36.
3. **The later source must read the earlier one.** Blocking attention from rule tokens to codebook tokens destroys rule reading in all four models; a size-matched sham block does not (Δ = −0.40 to −0.83). In Qwen3-4B, this dependency lives in layers 0–23 and runs through the codebook *value* tokens, not the key tokens.
4. **Out-of-scope application is model-dependent.** Qwen models rarely apply the rule when told not to (0–3 %). Phi-3.5-mini does so in 15–19 % of codebook-first cases; we confirm this on fresh data with a pre-registered test.

We also tested, with pre-registration, whether parameter-matched adapters that condition values on query–key pairs before aggregation use the rule more selectively than query-only, separable, or LoRA adapters. **They do not.** Every adapter learns the trained source order (0.85–0.99 at lr 5e-4) but transfers poorly to the unseen order (0.09–0.58), even though training on both orders reaches 0.69–0.92. Adaptation can introduce new out-of-scope application, and how much depends on the configuration more than on the operator class. Adapters reduce Phi's inherent leakage only partly, from 19 % to 6–11 % in codebook-first cases.

## 1 Introduction

Models are often given rules in context: formatting conventions, policies, procedures. A useful model must keep three behaviours apart. It must say what the rule is, apply it when the request authorizes it, and leave it alone when the request is out of scope. Failures in the third behaviour (scope leakage) and in the second (failure to apply) look alike from the outside but have different causes.

We isolate these behaviours with a synthetic task built for exact scoring:
- a **CODEBOOK** maps keys `x..` to values `z..`;
- a **RULE_MAP** P maps values to values (a derangement);
- an optional distractor rule Q competes with P;
- every query mode uses its own input key, so no answer can be copied across modes.

We score answers by teacher-forcing the JSON prefix and comparing the log-probabilities of four candidates: the gold answer, P(base), Q(base), and another value. This removes formatting errors, which dominated exact-match scoring in our pilots, where the model wrote correct content under the wrong key.

**Contributions.**
1. A behavioural dissociation in four frozen models. Rule *reading* is largely intact; rule *application* fails, mostly by stopping at the first hop; scope is usually respected, except in Phi-3.5.
2. A strong **source-order asymmetry**, and causal evidence that it arises because **the later source integrates the earlier one through attention**. In Qwen3-4B this integration is localized to the first two thirds of the network and to the codebook value tokens.
3. A **pre-registered negative result** for an architectural hypothesis: pairwise query–key conditioning of values before aggregation does not improve selective rule use over simpler adapters, either for composition or for repairing scope leakage.
4. Evidence that **adaptation does not transfer across source order**, even though the capacity exists, and that **adaptation can create scope leakage** to a degree that depends on the configuration.

## 2 Related work

- **In-context two-hop reasoning.** Guo et al. (2025)† study two-hop reasoning with distractors in 7–8B models and describe a sequential-query mechanism in small transformers. They keep premise order fixed and do not intervene on attention in pretrained LLMs. Latent multi-hop reasoning over *parametric* knowledge is studied by Press et al. (2022)†, Yang et al. (2024)†, Biran et al. (2024)† and Balesni et al.†
- **Order sensitivity.** Chen et al. (2024, *Premise Order Matters*)† report large drops when premises are out of reasoning order. Related work attributes order effects to the causal mask (*Repetition of Misordered Context*†; *Lost in the Prompt Order*†). We add a causal knockout with sham controls, plus a layer and token localization.
- **Attention gating and adapters.** Gated Attention (NeurIPS 2025) gates the SDPA output by the query. Dynamic Value Attention† and RCA† modulate values or gains per pair. Pairwise, feature-wise gates on messages before aggregation already appear in Gated GCN (Bresson & Laurent 2017), SAN and Point Transformer, and DiSAN. Our adapter is an instance of this family. We do not claim the operator as new; we test it as a frozen-LLM adapter against LoRA (Hu et al. 2022) and ReFT-style controls.

## 3 Task, models, and measurement

**Task.**
- Each bundle has 6 codebook entries, P, and Q.
- Modes: READ_DEC (decoded value), READ_RULE (P of a given decoded value; one hop), APPLY_P and APPLY_Q (two hops), and NONAPPLY (return the decoded value without applying P).
- Conditions: framing (neutral, or P described as a "standard procedure"), distractor (none, or Q present), and source order (codebook first `bf`, or rule first `rf`).
- Cue families: train/ID families and held-out OOD families.

**Models.** Frozen fp16 Qwen3-4B-Instruct-2507, Qwen3-1.7B, Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct.

**Probe.**
- The JSON prefix is teacher-forced, and candidates are scored by summed log-probability from the longest token prefix shared by all candidates.
- Metrics: choice accuracy, leakage (a NONAPPLY item answered with P(base)), stop-at-base, and the margin log p(gold) − log p(P(base)).

**Statistics.** Contrasts are paired by item. CIs are 95 % bootstrap intervals over bundle clusters (and seed × bundle where seeds exist).

**Pre-registration.** Every decision rule was written before its run: `docs/BENCHMARK_V2_SPEC.md`, `SCREEN_V2_PLAN.md`, `MECH_V2_PLAN.md`, `PAPER_A_PLAN.md`, `PHI_PLAN.md`. Post-hoc observations are labelled as such.

## 4 Frozen models: reading survives, application fails, order matters

*Table 1 (`t1_frozen.md`); Figure 1 (`f1_order_asymmetry`).*

| Model | READ_RULE bf | READ_RULE rf | APPLY bf | APPLY rf | stop-at-base | Leak bf | Leak rf |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.95 | 0.73 | 0.64 | 0.36 | 0.38 | 0/192 | 0/192 |
| Qwen3-1.7B | 0.87 | 0.48 | 0.19 | 0.04 | 0.84 | 7/192 | 5/192 |
| Qwen2.5-3B | 0.78 | 0.32 | 0.12 | 0.08 | 0.82 | 0/192 | 3/192 |
| Phi-3.5-mini | 0.80 | 0.38 | 0.66 | 0.02 | 0.63 | **29/192** | 0/192 |

**Reading versus applying.** With the first hop supplied, all models read the rule well (0.78–0.95 in bf). Applying it, which requires composing the codebook lookup with the rule lookup, is far worse, and the main error is to return the decoded value. In Qwen3-4B, 156 bundle-conditions read the rule correctly yet fail to apply it; the reverse happens in 25.

**Source order.** Rule-first order lowers READ_RULE in every model, by 0.22–0.46. It drives APPLY close to zero in three of the four models. Placing the codebook again right after the rule (Qwen3-4B) does not restore it: +0.08, CI [−0.02, 0.21]. Proximity to the answer therefore does not explain the effect.

**Scope.**
- Qwen models almost never apply the rule out of scope, even when P is framed as a default procedure. In Qwen3-4B the minimum log-prob margin is 17 nats.
- Phi-3.5-mini applies it in 15 % of codebook-first NONAPPLY items. We confirmed this on a fresh data seed with a pre-registered threshold (9/48 = 18.8 % ≥ 10 %).
- In Phi, leakage occurs only when the rule follows the codebook, which is also the only order in which Phi can apply the rule (APPLY 0.66 versus 0.02).

## 5 Mechanism: the later source must read the earlier one

*Table 2 (`t2_knockout_read_rule.md`); Figure 2 (`f2_knockout`); Table 3 (`t3_e2_narrow.md`).*

We add −∞ to the attention logits from one source span to another, in every layer and every head, during the prompt forward pass only. Answer positions still attend everywhere. A sham condition blocks an equal number of system-prompt tokens. A guard aborts the run unless the block reached every targeted layer.

**Effect across models.** Blocking rule → codebook (in bf order) removes rule reading in all four models, while the sham barely matters. Block − sham on READ_RULE:

| Model | Δ | CI95 |
|:---|---:|:---|
| Qwen3-4B | −0.71 | [−0.83, −0.58] |
| Qwen3-1.7B | −0.83 | [−0.94, −0.73] |
| Qwen2.5-3B | −0.54 | [−0.69, −0.40] |
| Phi-3.5-mini | −0.40 | [−0.56, −0.23] |

In rf order, blocking codebook → rule also hurts (Qwen3-4B: −0.31 on READ_RULE; −0.25 on APPLY). In both orders, therefore, the model accesses rule content *through* the later source's integration of the earlier one. In Phi, the same block also lowers leakage, from 31 % to 10 % in the neutral/none cell. Scope leakage thus runs through the same integration path.

**Localization (Qwen3-4B, bf).** Changes relative to no block:

| Rule tokens blocked from | ΔAPPLY | ΔREAD_RULE |
|:---|---:|---:|
| codebook, all layers | −0.62 | −0.75 |
| codebook, layers 0–11 | −0.62 | −0.33 |
| codebook, layers 12–23 | −0.62 | −0.69 |
| codebook, layers 24–35 | −0.02 | −0.02 |
| codebook *key* tokens only | −0.02 | −0.02 |
| codebook *value* tokens only | −0.62 | −0.54 |

**What this shows.** Rule tokens must see the codebook *value* tokens, the tokens shared between the two sources, and they must do so in the first two thirds of the network. They do not need the key tokens, so the full mapping `x → z′` is not pre-composed at the rule tokens. Our working account is a duplicate-token binding between the two sources, followed by completion at the answer position. Blocking only early layers hurts application more than reading (−0.62 versus −0.33), but no narrow block met our pre-registered criterion for separating composition from reading (ΔAPPLY ≤ −0.20 with ΔREAD ≥ −0.10). We do not claim a separate composition circuit.

## 6 Adaptation: learns the order it sees, does not fix scope selectively

*Table 4 (`t4_adaptation.md`); Figure 3 (`f3_adaptation`).*

**Setup.**
- Parameter-matched adapters on the frozen backbone: QUERY (query-only gate on a low-rank value branch); SEPARABLE (φ(q)⊙ψ(k)); VECTOR_PAIR (1+tanh(Qq+Kk) on each value before aggregation, reusing the native attention weights); and LoRA on q/k/v/o or on v alone.
- 256 steps on codebook-first items covering every mode. Evaluation includes the held-out rule-first order.

**No transfer across order (Qwen3-4B, 4 layers, 3 seeds).**
- When trained on codebook-first only, adapters reach 0.90–0.99 in distribution but 0.42–0.58 on rule-first order; the frozen model scores 0.33 there.
- Training on both orders reaches 0.69–0.92, so the capacity exists. What fails is generalization across order.
- In Phi, rule-first APPLY stays at ≤ 0.15 for every adapter.

**The architectural hypothesis (pre-registered, negative).**
- *Composition.* In single-seed screens, QUERY (+0.27 over frozen, 1 layer, lr 2e-3) and LoRA-qkvo (+0.19, 4 layers, lr 5e-4) improved rule-first application most. VECTOR_PAIR never beat the best control. In the 1-layer screen it was 0.15 below QUERY, CI [−0.29, 0.00].
- *Scope repair on Phi*, the one model with inherent leakage. In codebook-first cases all adapters lowered leakage from 18.8 % to:

  | Method | Leakage (bf) |
  |:---|---:|
  | QUERY | 11.1 % |
  | SEPARABLE | 10.4 % |
  | VECTOR_PAIR | 10.4 % |
  | LoRA-qkvo | 6.3 % |

  VECTOR_PAIR − LoRA = +0.08, CI [0.00, 0.17]. None removed the leakage, even though the training data included non-application items.
- *Leakage induced by adaptation (Qwen3-4B).*
  - With 4 layers, VECTOR_PAIR leaked least (0.6 %, against 3.0 % for SEPARABLE and 9.6 % for LoRA-qkvo; paired CIs exclude 0).
  - With 1 layer, the ranking reversed for VECTOR_PAIR: QUERY 0.7 % < SEPARABLE 2.8 % < VECTOR_PAIR 6.3 % < LoRA-v 10.4 %.
  - Following our pre-registered reading, we treat the operator ranking as configuration-dependent. LoRA leaked most in both Qwen configurations.
  - A rerun of the single-seed screen on different hardware reproduced the VECTOR_PAIR count (14 versus 15/96) but not the QUERY count (0 versus 7/96). Leakage measured from a few items is noisy, which is why we report only multi-seed aggregates.

## 7 Discussion and limitations

- **Scope of the claims.** Our claims cover small models (1.7–4B) from two families, one synthetic task family, English templates, and log-probability scoring.
- **Mechanism.** The account is supported by necessity tests (knockouts). It does not yet have sufficiency tests (patching) or head-level localization.
- **Adapters.** Adapter results depend on placement and learning rate, and we explored these only partly (one revisit, two learning rates).
- **Leakage in Phi.** Leakage in Phi is lower under default framing (0–4 %) than under neutral framing (6–19 %). We cannot explain this yet.
- **Takeaway for practitioners.** When a model must apply in-context rules to looked-up values, place the lookup source before the rule. Fine-tuning on one layout should not be expected to transfer to another.

## 8 Reproducibility

- **Receipts and integrity.** Every run stores per-item receipts with checksums, a config fingerprint, the model revision and the source hash (`deploy/EXECUTION.json`).
- **Notebooks.** Notebooks are plain text, with no encoded payloads.
- **Pipeline.** `paper/analysis.py` regenerates all tables and figures. Pre-registrations are in `docs/`.
- **Compute.** About 2.5 GPU-hours on T4, plus about 12 Colab compute units on an RTX PRO 6000.

---

## Before submission: missing work, ranked

1. **Sufficiency and localization.** Activation patching from book-first into rule-first runs, plus head-level attention analysis of the duplicate-token binding (layers 0–11). This is the main gap for a mechanism paper.
2. **External validity.**
   - A natural-language version of the task (procedures and policies written in prose), with ~100 items.
   - A free-generation check that the log-prob conclusions hold under greedy decoding (a subset).
3. **Scale.** One 7–8B model (Qwen3-8B, or Llama-3.1-8B if an HF license token is available) for the order asymmetry and the knockout.
4. **Leakage statistics.** 5 seeds for the adaptation-leakage configurations, and an lr sweep at a fixed layer count.
5. **Phi framing anomaly.** Why does default framing reduce leakage?
6. **Citations.** Read the full PDFs of Guo et al. 2025 and the order-sensitivity papers; verify every † claim.
