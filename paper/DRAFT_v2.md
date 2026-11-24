# Binding Before Reading: A Two-Stage Attention Circuit for Applying In-Context Rules, and Why Source Order Matters Only in Structured Prompts

*Draft v2, 23 Oct 2026. Every number is produced by `paper/analysis.py` (layout `paper/layout_original.json`) from per-item receipts. Tables are in `paper/tables/`, figures in `paper/figures/`, and citation keys refer to `paper/refs.bib`. Pre-registrations are in `docs/` (v2-spec, screen, mech, PAPER_A, PHI, ROUND3, ROUND4). The text marks post-hoc analyses explicitly.*

## Abstract

When a model is given a lookup table and a rule in context, it may be asked to *read* the rule, to *apply* it (a two-hop composition), or to leave it alone because the request is out of scope. We study these behaviours in **nine frozen instruction-tuned models from four families** (Qwen3 1.7–14B, Qwen2.5-3B, Phi-3.5-mini, OLMo-2-7B, Granite-3.3-8B). Scoring uses teacher-forced candidate log-probabilities, which removes formatting confounds; in free generation, small models answer apply requests with the read schema.

1. **Reading survives, applying fails.** Given the first hop, models read the rule well. When they must compose the two hops themselves, they mostly stop after the first. This holds in all nine models.
2. **A two-stage attention circuit.** Attention knockouts with size-matched sham controls show that the rule can only be used if the *later* source attends to the *shared value tokens* of the earlier one. This happens in early and middle layers and is concentrated in a few duplicate-token **binding heads**:
   - blocking 8 of them removes application;
   - blocking them also cuts **reader heads'** attention, at the answer position, to the queried rule entry by 65–75 %, while random heads change nothing;
   - restoring clean rule-token states in a mid-layer window brings rule reading back.
3. **Source order matters only in the structured format.** Putting the rule before the codebook lowers rule reading by 0.18–0.46 in all nine models with the JSON codebook. Across three prose templates the asymmetry disappears, although **the same binding heads** are used (Jaccard 0.49–0.73 against 0.02 for random sets). What differs is how well the rule-first JSON prompt supports the integration, not which circuit runs.
4. **Scope leakage is model- and format-dependent.** Some models apply the rule out of scope (Phi-3.5: 15–32 % codebook-first), and prose increases leakage in weak readers. Describing the rule explicitly as a procedure *reduces* leakage, while a content-free sentence in the same slot does not.

**Adapters (pre-registered).** Parameter-matched adapters that gate each value by its query–key pair do **not** use the rule more selectively than query-only, separable or LoRA adapters. No adapter transfers across source order, although training on both orders shows the capacity exists. Under aggressive training, LoRA introduces more scope leakage than value-branch adapters (5 seeds).

## 1 Introduction

Models are routinely given rules in context: policies, formatting conventions, procedures. Using a rule well involves three different behaviours:
- stating it;
- applying it when authorized, which requires composing a lookup with the rule;
- withholding it when the request is out of scope.

Failures in application and in scope look similar from the outside but have different causes. We ask **where** in the network a rule becomes usable, **why** the layout of the context changes that, and whether lightweight adapters can make rule use more selective.

**Contributions.**
1. A behavioural dissociation between reading and applying in nine models from four families, measured format-free.
2. **A causal two-stage circuit:**
   - *binding heads*, where later-source value tokens attend to the same values in the earlier source;
   - *reader heads*, where the answer position attends to the queried rule entry.
   - Evidence: necessity, partial sufficiency, a restoration window, and a binding → reader path effect.
3. **Format dependence:** the order asymmetry is specific to the JSON encoding; the same heads operate in prose.
4. A pre-registered **negative result** for pairwise query–key value-gating adapters, plus two adaptation findings: no transfer across order, and LoRA-specific scope leakage under aggressive training.

## 2 Related work

- **In-context two-hop reasoning.** [guo2025twohop] show that pretrained LLMs given distractors fall back to random guessing among plausible conclusions, that a few fine-tuning steps fix this, and they analyse the mechanism in a small transformer trained from scratch.‡ We localize a circuit *in pretrained LLMs* and vary the source order.
- **Parametric multi-hop.** Studied by [press2023compositionality], [yang2024latent], [biran2024hopping] (the bridge entity resolves early and the second hop starts too late) and [balesni2024twohop] (composition fails for two synthetic facts but succeeds when one is natural).
- **Order sensitivity.** [chen2024premise] report drops of more than 30 % when premises are permuted. [yu2025core] show that document order matters and that repeating the context helps. [ok2026lost] attribute a context-first gap to the causal mask. We show that the order effect depends on the format and identify the integration step it acts on.
- **Attention gating and adapters.** [qiu2025gated], [wang2025dva], [zhao2026rca]. Pairwise feature-wise gating on messages already exists in [bresson2017rggc], [zhao2020san], [zhao2020pointtransformer] and [shen2018disan]. Baselines: LoRA [hu2021lora], ReFT [wu2024reft], and head steering [zhang2024pasta].

‡ Checked against the abstract; the claims that premise order is fixed and that no knockout is done in pretrained models still need a check against the full text.

## 3 Task, models, measurement

**Task.**
- A codebook of 6 entries (keys → values), a rule P (a derangement of the values), and an optional distractor rule Q.
- Each mode uses its own input key. Modes: READ_DEC; READ_RULE (P of a supplied decoded value, one hop); APPLY_P and APPLY_Q (two hops); NONAPPLY.
- Conditions: framing (neutral, default procedure, content-free filler), distractor, and order (codebook first `bf` or rule first `rf`).
- Formats:
  - **JSON codes** (`x..` → `z..`);
  - **three prose templates**: staff directory → rooms; flight board → gates; inventory → shelves. Each has prose procedures and keeps the same JSON answer contract.

**Probe.** The JSON prefix is teacher-forced, and four candidates are scored by summed log-probability from their longest shared token prefix.

**Interventions.**
- *Attention knockout:* −∞ on the logits from one span to another during the prompt pass, per layer and optionally per head. A size-matched sham blocks the same number of system-prompt tokens. A guard verifies that every targeted layer or head was hit.
- *Restoration:* write back clean later-source states after layer L.

**Statistics.** Paired cluster bootstraps (95 %). Pre-registrations are noted in each section.

## 4 Behaviour across nine models

### 4.1 Reading versus applying, and source order (JSON codes)

*Tables `t1_frozen`, `t7_scale_frozen`, `t12_families_frozen`, `t12c`.*

| Model | READ_RULE bf | READ_RULE rf | APPLY bf | APPLY rf | stop-at-base | Leak (bf) |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.87 | 0.48 | 0.19 | 0.04 | 0.84 | 7/192 |
| Qwen2.5-3B | 0.78 | 0.32 | 0.12 | 0.08 | 0.82 | 0/192 |
| Phi-3.5-mini | 0.80 | 0.38 | 0.66 | 0.02 | 0.63 | 29/192 |
| Qwen3-4B | 0.95 | 0.73 | 0.64 | 0.36 | 0.38 | 0/192 |
| OLMo-2-7B | 0.71 | 0.32 | 0.12 | 0.10 | 0.83 | 1/192 |
| Qwen3-8B | 0.95 | 0.76 | 0.47 | 0.26 | 0.43 | 1/192 |
| Granite-3.3-8B | 0.70 | 0.47 | 0.27 | 0.13 | 0.75 | 6/192 |
| Qwen3-14B | 0.95 | 0.61 | 0.81 | 0.49 | 0.16 | 0/192 |

In every model, reading exceeds applying, and codebook-first exceeds rule-first for reading. For the three newest models (OLMo, Granite, Qwen3-14B), all three pre-registered contrasts have CIs excluding 0 (`t12c`).

### 4.2 Prose: the order asymmetry disappears

*Tables `t6b_nl_frozen`, `t12d`, `t13_*`; Figure 6.*

Across all three prose templates the codebook-first minus rule-first gap in READ_RULE is near 0; |Δ| < 0.10 or the CI includes 0 in 4 of 5 models for each new template. The one exception is Phi on the gate template, where the gap *reverses* (−0.21).

The read-over-apply gap persists in prose for strong prose readers: Qwen3-4B (+0.40 to +0.77), Qwen3-8B, and Phi on shelves. Weak prose readers (OLMo, Granite) are near floor.

### 4.3 Scope

- **Qwen3-4B and Qwen3-14B** never apply the rule out of scope.
- **Phi-3.5** does so in 15–32 % of codebook-first items; this was confirmed on fresh data with a pre-registered threshold.
- **Prose** increases leakage in weak readers (OLMo up to 63/384).
- **Framing (pre-registered control).** Describing P as the standard procedure lowers Phi's leakage (32 % → 18 %); a content-free sentence in the same slot does not (31 %). Difference −0.14 [−0.22, −0.05].

### 4.4 Free generation

Generations that parse agree with the log-prob choice in 90–100 % of cases. However, 26–67 % of generations do not parse into a candidate, mostly because apply and non-apply requests get the read schema. Free generation therefore cannot measure the apply decision in small models.

## 5 A two-stage circuit

### 5.1 Necessity

*`t2`, `t7b`, `t12b`; Figure 2.* Blocking rule → codebook attention (bf) removes rule reading in all nine models; the sham barely matters. Block − sham on READ_RULE ranges from −0.40 (Phi) to −0.83 (Qwen3-1.7B); OLMo −0.75, Granite −0.50, Qwen3-14B −0.67.

### 5.2 Binding heads

*Qwen3-4B, `t3`, `t5`.*
- **Tokens:** blocking only the codebook value tokens is as damaging as blocking the whole codebook; blocking the key tokens does nothing.
- **Layers:** blocking layers 24–35 does nothing.
- **Heads:** the top-8 duplicate-token heads, which lie in layers 1–19, account for the effect. Blocking them gives APPLY 0.04 against 0.61 for random-8, and READ 0.56 against 0.94. The same holds for Phi (−0.51 and −0.23).
- **Sufficiency (`t10`).** Letting *only* the top-32 binding heads read the codebook restores READ to 0.73–0.75 (clean 0.90–0.96; all blocked 0.21–0.48). That is 64–72 % of the effect. It does not restore APPLY, so binding is concentrated in a few heads but not confined to 32 (pre-registered P2a not met).

### 5.3 Reader heads and the path

*`t10`.*
- **Reader heads.** We score each head by its attention from the answer position to the queried rule entry. Blocking the top-8 reader heads lowers APPLY by 0.34–0.68 more than random-8, in all three models tested. READ_RULE is barely affected; its prefix already contains the decoded value, so it uses a different route.
- **Path (pre-registered P2c, met).** Blocking the top-8 *binding* heads lowers the reader heads' attention to the queried entry from 0.42 to 0.11 (Qwen3-4B), 0.37 to 0.11 (Qwen3-8B), and 0.53 to 0.18 (Phi). Random-8 blocks leave it unchanged (Δ −0.26 to −0.34; CIs exclude 0).
- **The application circuit:** binding heads attach each rule entry to the matching codebook value, and reader heads at the answer position then find the entry for the queried value.

### 5.4 Restoration window

*`t5`, `t11`; Figures 4–5.*
- Restoring clean later-source states after layer L brings rule reading back **only within a mid-layer window**: L ≈ 14–20 in Qwen3-4B/8B and L ≈ 12–18 in Phi. The window is the same for both formats and both orders.
- **Our account (post-hoc).** Integration is complete by the start of the window, and positions after the source read it only after the window.

## 6 Why order matters only in JSON

*`t11`, `t11b`; Figure 5.*

**Same heads.** The top-32 binding heads overlap strongly across formats and orders (Jaccard 0.49–0.73; random 0.02). Prose and JSON therefore do not use different heads.

**Same restoration window.** In codes rule-first, restoration recovers close to that order's *clean* level (Qwen3-4B 0.75 vs 0.79). The deficit lies in the clean computation, not in a missing mechanism.

**What differs is how much accuracy depends on the integration** (clean − blocked READ, post-hoc):
- in JSON it is asymmetric: −0.75 when the rule comes second versus −0.31 when the codebook comes second (Qwen3-4B);
- in prose it is nearly symmetric: −0.52 versus −0.43.

So when the codebook comes second in JSON, the integration contributes less and the model is less accurate. In prose, the explicit relational frame of every entry ("badge K is assigned to room V"; "room V moves to room W") lets the later source integrate the earlier one equally well in both orders. This matches our pre-registered hypothesis H-prose (P1a partly met, P1b met).

**Caveat.** In prose, Qwen3-8B depends little on this route (blocked 0.65 vs clean 0.75; the prose knockout is not significant), which points to a second route at larger scale.

## 7 Adaptation (pre-registered; Qwen3-4B and Phi; 5 seeds)

*`t4`, `t4b`; Figure 3.*

**Setup.** Parameter-matched adapters on frozen backbones: QUERY; SEPARABLE φ(q)⊙ψ(k); VECTOR_PAIR 1+tanh(Qq+Kk) on each value before aggregation; LoRA on q/k/v/o or on v. 256 steps on codebook-first items.

- **No transfer across order.** Trained on codebook-first only, adapters reach 0.90–0.96 in distribution but 0.43–0.63 rule-first (frozen 0.33). Training on both orders reaches 0.70–0.94. In Phi, rule-first APPLY stays ≤ 0.14.
- **The architectural hypothesis is negative.**
  - VECTOR_PAIR never beat the best control at composition.
  - It did not reduce Phi's leakage more than the others. Codebook-first leakage goes from 18.8 % frozen to 7.9–11.7 % after adaptation, with LoRA lowest.
- **Leakage induced by adaptation.**
  - Under aggressive training (lr 2e-3, or training on both orders), LoRA-qkvo leaks more than Separable and Vector pair, by 0.07–0.10 with CIs excluding 0.
  - Under mild training there are no differences.
  - With one layer, Separable leaks least, and the pairwise component never helps.

## 8 Discussion and limitations

**What generalizes:**
- the reading/application dissociation;
- the necessity of later → earlier integration through shared value tokens;
- a two-stage binding → reader circuit (shown in 3 models);
- the JSON-specific order asymmetry (9 models, 4 families, up to 14B).

**What does not:**
- the order asymmetry in prose (3 templates);
- the prose knockout in Qwen3-8B;
- clean read/apply contrasts in weak prose readers.

**Mechanism limits:**
- binding is not sufficient with 32 heads;
- reader heads explain application but not one-hop reading;
- the restoration window is interpreted post hoc;
- there is no full path patching through MLPs.

**Practical takeaways:**
- in structured prompts, put the lookup table before the rule;
- prose descriptions of rules are order-robust;
- fine-tuning on one layout does not transfer to another;
- prefer value-branch adapters over LoRA when scope must be preserved under aggressive adaptation.

## 9 Reproducibility

- **Code release** (`README.md`): generator, probes, knockouts (per layer and per head), restoration, binding and reader head analysis, adapters, a parallel single-GPU job runner that retries failed stages, a full reproduction spec (`tools/make_job_specs.py`, 96 stages), and `paper/analysis.py`.
- **Raw receipts:** in a separate archive, identified by SHA256.
- **Compute:** about 2.5 T4 GPU-hours plus about 36 Colab compute units on an RTX PRO 6000.
