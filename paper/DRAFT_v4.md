# Binding Rewrites the Address: A Key-Mediated Attention Circuit for Applying In-Context Rules, and Why Source Order Matters Only in Structured Prompts

*Draft v4, 23 Oct 2026. Every number is produced by `paper/analysis.py` (layout `paper/layout_original.json`) from per-item receipts. Tables are in `paper/tables/`, figures in `paper/figures/`, and citation keys refer to `paper/refs.bib`. Pre-registrations are in `docs/` (v2-spec, screen, mech, PAPER_A, PHI, ROUND3, ROUND4, ROUND5, ROUND6). The text marks post-hoc analyses explicitly.*

## Abstract

When a model is given a lookup table and a rule in context, it may be asked to *read* the rule, to *apply* it (a two-hop composition), or to leave it alone because the request is out of scope. We study these behaviours in **fourteen frozen instruction-tuned models from seven families**: Qwen3 1.7–14B, Qwen2.5-3B, Phi-3.5-mini, OLMo-2-7B, Granite-3.3-8B, Gemma-2 2B/9B, Mistral-7B and Llama-3 3B/8B. Scoring uses teacher-forced candidate log-probabilities, which removes formatting confounds; in free generation, small models answer apply requests with the read schema.

1. **Reading survives, applying fails.** Given the first hop, models read the rule well. When they must compose the two hops themselves, they mostly stop after the first. This holds in every model that can read the rule (13 of 14).
2. **A two-stage attention circuit that works through keys.** Attention knockouts with size-matched sham controls show that the rule can be used only if the *later* source attends to the *shared value tokens* of the earlier one. This integration is written by attention heads rather than MLPs, mostly by a few duplicate-token **binding heads**, and what it changes is the **keys** of the rule tokens:
   - **Binding heads → reader heads.** Blocking 8 binding heads removes application. In 6 models from 5 families it also cuts by 65–94 % the attention that **reader heads** at the answer position pay to the queried rule entry; blocking random heads changes nothing.
   - **Keys suffice.** With the integration blocked everywhere, restoring only the clean **keys** at the rule tokens, in the 6–8 key/value groups that hold the top reader heads, brings rule reading back in all 6 models tested, on two independent sets of bundles. Restoring their values does nothing.
   - **The reader heads themselves carry it.** Giving clean keys to the top reader *query heads* alone recovers 73–116 % of the group-level effect in 4 of 5 grouped-query models, while their group-mates recover nothing. In Qwen3-8B, the keys seen by a single head recover half of it.
   - **Restoration window.** Restoring clean rule-token states in a mid-layer window also brings rule reading back.
3. **Source order matters only in the structured format.**
   - With the JSON codebook, putting the rule before the codebook lowers rule reading by 0.16–0.61 in all fourteen models.
   - Across three prose templates this asymmetry disappears, although **the same binding heads** are used (Jaccard 0.49–0.73, against 0.02 for random sets).
   - In prose, a rule entry can also be found by matching the queried word directly. With the codebook removed, reader heads keep 25–76 % of their attention to the right entry, against 5–19 % in JSON. A *partial* block, which leaves a corrupted binding, hurts more than removing the codebook altogether.
4. **Scope leakage depends on the model and the format.** Some models apply the rule out of scope (Phi-3.5: 15–32 % with the codebook first), and prose increases leakage in weak readers. Describing the rule explicitly as a procedure *reduces* leakage, while a content-free sentence in the same slot does not.

**Adapters (pre-registered).**
- Parameter-matched adapters that gate each value by its query–key pair do **not** use the rule more selectively than query-only, separable or LoRA adapters.
- No adapter transfers across source order, although training on both orders shows that the capacity exists.
- Under aggressive training, LoRA introduces more scope leakage than value-branch adapters (5 seeds).

## 1 Introduction

Models are routinely given rules in context: policies, formatting conventions, procedures. Using a rule well involves three different behaviours:
- stating it;
- applying it when authorized, which requires composing a lookup with the rule;
- withholding it when the request is out of scope.

Failures of application and of scope look similar from the outside but have different causes. We ask **where** in the network a rule becomes usable, **through which interface** the answer position reaches it, **why** the layout of the context changes that, and whether lightweight adapters can make rule use more selective.

**Contributions.**
1. **A behavioural dissociation** between reading and applying, measured without formatting confounds, in fourteen models from seven families.
2. **A causal two-stage circuit, with the interface between its stages identified:**
   - *binding heads*, where later-source value tokens attend to the same values in the earlier source. Their direct contributions write the binding; the MLPs' do not.
   - *reader heads*, where the answer position attends to the queried rule entry.
   - The interface is the **keys** of the rule tokens as seen by individual reader query heads: restoring only those keys is enough for rule reading.
   - The evidence covers necessity, partial sufficiency, sender decomposition, receiver key/value restoration, a restoration window and a binding → reader path effect. Heads for the key/value and decomposition tests are selected on held-out bundles.
3. **Format dependence:** the order asymmetry is specific to the JSON encoding. The same heads operate in prose, and prose also supports a direct-match route.
4. **A pre-registered negative result** for pairwise query–key value-gating adapters, plus two adaptation findings: no transfer across order, and LoRA-specific scope leakage under aggressive training.

## 2 Related work

- **In-context two-hop reasoning.** [guo2025twohop] find that pretrained LLMs given distractors fall back to random guessing among plausible conclusions, that a few fine-tuning steps fix this, and they analyse the mechanism in a small transformer trained from scratch.‡ We localize a circuit *in pretrained LLMs*, identify the interface between its stages, and vary the source order.
- **Parametric multi-hop.** Studied by [press2023compositionality], [yang2024latent], [biran2024hopping] (the bridge entity resolves early and the second hop starts too late) and [balesni2024twohop] (composition fails for two synthetic facts but succeeds when one of them is natural).
- **Order sensitivity.** [chen2024premise] report drops of more than 30 % when premises are permuted. [yu2025core] show that document order matters and that repeating the context helps. [ok2026lost] attribute a context-first gap to the causal mask. We show that the order effect depends on the format, and identify the integration step it acts on.
- **Attention gating and adapters.** [qiu2025gated], [wang2025dva], [zhao2026rca]. Pairwise feature-wise gating of messages already exists in [bresson2017rggc], [zhao2020san], [zhao2020pointtransformer] and [shen2018disan]. Baselines: LoRA [hu2021lora], ReFT [wu2024reft] and head steering [zhang2024pasta].

‡ Checked against the abstract only. Two claims still need checking against the full text: that premise order is fixed, and that no knockout is done in pretrained models.

## 3 Task, models, measurement

**Task.**
- A codebook of 6 entries (keys → values), a rule P (a derangement of the values) and an optional distractor rule Q.
- Each mode uses its own input key. The modes are:
  - READ_DEC: decode the input key;
  - READ_RULE: P of a supplied decoded value (one hop);
  - APPLY_P and APPLY_Q: two hops;
  - NONAPPLY: out of scope.
- Conditions: framing (neutral, default procedure, or a content-free filler), distractor, and order (codebook first, `bf`, or rule first, `rf`).
- Formats:
  - **JSON codes** (`x..` → `z..`);
  - **three prose templates**: staff directory → rooms; flight board → gates; inventory → shelves. Each states its procedures in prose and keeps the same JSON answer contract.
- Gemma-2's chat template has no system role, so the contract opens the user turn.
- Llama-3 is loaded from the ungated `unsloth` mirrors of the official checkpoints, which hold the same weights; the Llama 3.1/3.2 Community License applies. Llama-3.2's template writes the current date into the system turn; we pin it to Llama-3.1's default so that prompts do not depend on the run date.

**Probe.** The JSON prefix is teacher-forced. Four candidates are scored by their summed log-probability from the longest token prefix they share.

**Interventions.**
- *Attention knockout:* −∞ on the attention logits from one span to another during the prompt pass, per layer and optionally per head. A size-matched sham blocks the same number of system-prompt tokens instead. A guard verifies that every targeted layer or head was hit.
- *Restoration:* write back clean later-source states after layer L.
- *Sender decomposition:* add back selected components' clean-minus-blocked direct contributions to the residual stream (§5.4).
- *Receiver restoration:* give selected key/value groups, or single query heads, their clean keys and/or values (§5.5–5.6).

**Statistics.** Paired cluster bootstraps (95 %). Pre-registrations are noted in each section.

## 4 Behaviour across fourteen models

### 4.1 Reading versus applying, and source order (JSON codes)

*Tables `t1_frozen`, `t7_scale_frozen`, `t12_families_frozen`, `t12c`, `t17`, `t17c`.*

| Model | READ_RULE bf | READ_RULE rf | APPLY bf | APPLY rf | stop-at-base | Leak (bf) |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.87 | 0.48 | 0.19 | 0.04 | 0.84 | 7/192 |
| Llama-3.2-3B | 0.17 | 0.01 | 0.15 | 0.02 | 0.89 | 8/192 |
| Gemma-2-2B | 0.71 | 0.10 | 0.06 | 0.01 | 0.96 | 5/192 |
| Qwen2.5-3B | 0.78 | 0.32 | 0.12 | 0.08 | 0.82 | 0/192 |
| Phi-3.5-mini | 0.80 | 0.38 | 0.66 | 0.02 | 0.63 | 29/192 |
| Qwen3-4B | 0.95 | 0.73 | 0.64 | 0.36 | 0.38 | 0/192 |
| Mistral-7B | 0.65 | 0.45 | 0.10 | 0.13 | 0.84 | 0/192 |
| OLMo-2-7B | 0.71 | 0.32 | 0.12 | 0.10 | 0.83 | 1/192 |
| Qwen3-8B | 0.95 | 0.76 | 0.47 | 0.26 | 0.43 | 1/192 |
| Granite-3.3-8B | 0.70 | 0.47 | 0.27 | 0.13 | 0.75 | 6/192 |
| Llama-3.1-8B | 0.73 | 0.35 | 0.57 | 0.22 | 0.52 | 17/192 |
| Gemma-2-9B | 0.92 | 0.71 | 0.29 | 0.15 | 0.76 | 0/192 |
| Qwen3-14B | 0.95 | 0.61 | 0.81 | 0.49 | 0.16 | 0/192 |

Qwen3-1.7B, Qwen2.5-3B, Phi-3.5-mini and Qwen3-4B were tested from the start; OLMo, Granite and Qwen3-14B were added in round 4, Gemma-2 and Mistral in round 5, and Llama-3 in round 6.
- **In every model that can read the rule**, reading exceeds applying; in all fourteen, reading is better with the codebook first than with the rule first. Llama-3.2-3B barely reads the rule (0.17), so its read/apply contrast is uninformative.
- **Round-4 models** (OLMo, Granite, Qwen3-14B): all three pre-registered contrasts have CIs excluding 0 (`t12c`).
- **Round-5 models** (Gemma-2 2B/9B, Mistral-7B; `t17c`): read > apply and bf > rf hold in all three, and the knockout contrast holds in two. Mistral is the exception (−0.15 [−0.31, +0.02]); its sham block alone already costs 0.16.
- **Round-6 model Llama-3.1-8B** (`t17c`): all three contrasts hold (+0.16, +0.38, −0.69).

### 4.2 Prose: the order asymmetry disappears

*Tables `t6b_nl_frozen`, `t12d`, `t13_*`, `t17d`; Figure 6.*

- **No order gap in prose.** Across all three prose templates, the READ_RULE gap between codebook first and rule first is near 0: |Δ| < 0.10, or the CI includes 0, in 4 of 5 models for each new template.
  - The one exception is Phi on the gate template, where the gap *reverses* (−0.21).
  - Gemma-2-9B, a strong prose reader, shows no gap either (0.83 vs 0.88).
- **The read-over-apply gap persists in prose for strong prose readers:** Qwen3-4B (+0.40 to +0.77), Qwen3-8B, and Phi on shelves. Weak prose readers (OLMo, Granite, Gemma-2-2B, both Llama-3 models) are near floor.

### 4.3 Scope

- **Qwen3-4B, Qwen3-14B and Gemma-2-9B** never apply the rule out of scope.
- **Phi-3.5** does so in 15–32 % of codebook-first items. This was confirmed on fresh data with a pre-registered threshold.
- **Prose** increases leakage in weak readers (OLMo up to 63/384; Mistral 23/192 with the codebook first).
- **Llama-3.1-8B** leaks in 17/192 codes items with the codebook first.
- **Framing (pre-registered control).** Describing P as the standard procedure lowers Phi's leakage from 32 % to 18 %. A content-free sentence in the same slot does not (31 %). The difference is −0.14 [−0.22, −0.05].

### 4.4 Free generation

Generations that parse agree with the log-prob choice in 90–100 % of cases. However, 26–67 % of generations do not parse into a candidate, mostly because apply and non-apply requests get the read schema. Free generation therefore cannot measure the apply decision in small models.

## 5 A two-stage circuit

### 5.1 Necessity

*`t2`, `t7b`, `t12b`, `t17b`; Figure 2.*

- **The block removes rule reading in thirteen of fourteen models,** and the sham barely matters. The block is rule → codebook attention with the codebook first.
- **Block − sham on READ_RULE** ranges from −0.33 (Gemma-2-9B) to −0.83 (Qwen3-1.7B): OLMo −0.75, Gemma-2-2B −0.73, Llama-3.1-8B −0.69, Qwen3-14B −0.67, Granite −0.50. In the weak reader Llama-3.2-3B it is −0.10 [−0.19, −0.02].
- **Mistral-7B is the exception.** The sham itself costs 0.16, and the difference is −0.15 [−0.31, +0.02].

### 5.2 Binding heads

*Qwen3-4B, `t3`, `t5`; sufficiency `t10`, `t18`.*

- **Tokens.** Blocking only the codebook value tokens is as damaging as blocking the whole codebook. Blocking the key tokens does nothing.
- **Layers.** Blocking layers 24–35 does nothing.
- **Heads.** The top-8 duplicate-token heads, which lie in layers 1–19, account for the effect.
  - Blocking them gives APPLY 0.04, against 0.61 for random-8, and READ 0.56 against 0.94.
  - The same holds for Phi (−0.51 and −0.23).
- **Sufficiency.** Here *only* the top-32 binding heads may read the codebook.
  - In Qwen3-4B/8B and Phi, READ comes back to 0.73–0.75 (clean 0.90–0.96; all blocked 0.21–0.48), i.e. 64–72 % of the effect.
  - In Gemma-2-9B it comes back to within 0.06 of clean (0.85 vs 0.92); in Llama-3.1-8B hardly at all (0.12 vs 0.65), the most distributed binding we see.
  - APPLY does not come back. So binding is concentrated in a few heads but not confined to 32 (pre-registered P2a not met).

### 5.3 Reader heads and the path

*`t10`, `t18`.*

- **Reader heads.** We score each head by its attention from the answer position to the queried rule entry.
  - Blocking the top-8 reader heads lowers APPLY by 0.30–0.68 more than random-8 in Qwen3-4B/8B, Phi, Gemma-2-9B and Llama-3.1-8B (−0.48).
  - Mistral's APPLY is at floor; there, READ drops by 0.34.
- **Path (pre-registered P2c, met in 6/6).** Blocking the top-8 *binding* heads lowers the reader heads' attention to the queried entry:

  | Model | Clean | Top-8 binding heads blocked |
  |:---|---:|---:|
  | Qwen3-4B | 0.42 | 0.11 |
  | Qwen3-8B | 0.37 | 0.11 |
  | Phi-3.5-mini | 0.53 | 0.18 |
  | Gemma-2-9B | 0.33 | 0.07 |
  | Mistral-7B | 0.18 | 0.01 |
  | Llama-3.1-8B | 0.28 | 0.03 |

  Blocking 8 random heads leaves it unchanged (Δ −0.17 to −0.34; CIs exclude 0).

### 5.4 Who writes the binding (sender side, pre-registered G5)

*`t14a–c`; Figure 7. Fresh bundles; heads ranked on held-out bundles.*

**Method.** Rule → codebook attention is blocked in every layer. After layer L*, at the rule tokens, we add back the clean-minus-blocked **direct contributions** of a chosen set of components from layers 0..L*:
- a head's contribution is its slice of the `o_proj` input times its slice of `o_proj`;
- an MLP's contribution is its output.

With all components, this rebuilds the clean residual exactly (in pre-norm decoders) and recovers 63–92 % of the READ_RULE gap. The remainder is the restoration ceiling at L*, since later layers stay blocked.

**Results.**
- **Attention, not MLPs (P5b met).** All heads' contributions together recover 32–57 % of the gap; all MLPs' recover 19 % or less. Attention − MLP is +0.12 to +0.31, with CIs excluding 0 in 3 of 4 models.
- **Concentration varies (P5c not met).**
  - The top-32 binding heads recover 0.51 of the gap in Qwen3-4B (+0.33 over random-32) and 0.21 in Qwen3-8B (+0.15).
  - In Phi and Mistral they recover little more than random heads.
- **Not monotonic in k.** Adding the top-128 heads' contributions recovers *less* than the top-32; in Qwen3-4B it falls below the blocked level. Partial direct-path patches create residual states that later layers do not handle. Beyond the strongest heads, the binding is distributed and has to be restored jointly.
- **MLPs do not explain why 32 heads are insufficient (P5d).** Top-32 heads plus all MLPs is no better than the top-32 alone: −0.08 to +0.10, all CIs include 0.
- **Llama-3.1-8B (round 6).** The decomposition is uninformative there: restoring the full residual at the pre-registered L* = 14 recovers almost nothing (0.15 vs 0.85 clean, 0.02 blocked), so L* lies outside Llama's window.

### 5.5 How the readers use it: keys, not values (receiver side, pre-registered G6)

*`t15`; Figure 8.*

**Method.**
- Start from the fully blocked run.
- At the rule tokens, give clean keys and/or values only to the key/value groups that hold the top-8 reader heads. That is 6–8 of 288–336 groups; in grouped-query models one group serves 2–4 query heads.
- The control does the same for the groups of 8 random heads.

| Model (codes) | READ clean / blocked | K+V / K only / V only / random K+V | K − V |
|:---|:---|:---|:---|
| Qwen3-4B | 0.96 / 0.19 | 0.77 / 0.75 / 0.21 / 0.33 | +0.54 [+0.40, +0.69] |
| Qwen3-8B | 1.00 / 0.21 | 0.79 / 0.71 / 0.25 / 0.34 | +0.46 [+0.31, +0.60] |
| Phi-3.5-mini | 0.92 / 0.52 | 0.77 / 0.81 / 0.46 / 0.53 | +0.35 [+0.19, +0.50] |
| Mistral-7B | 0.60 / 0.17 | 0.46 / 0.38 / 0.15 / 0.22 | +0.23 [+0.10, +0.35] |
| Gemma-2-9B | 0.94 / 0.69 | 0.88 / 0.88 / 0.71 / 0.72 | +0.17 [+0.06, +0.27] |

- **Rule reading comes back through the keys.**
  - Restoring K+V beats random K+V by 0.16–0.45 in all five models. On fresh bundles (round 6, `t19b`), it does so by 0.26–0.54 in all six models including Llama-3.1-8B, and keys alone beat values alone in 5 of 6 (Mistral's lower bound is 0.00).
  - Restoring the keys alone does as well as restoring keys and values.
  - Restoring the values alone does nothing. K − V > 0 with CIs excluding 0 in all five models, and in 3 of 4 prose runs.
- **Application (P6a not met, 2/5).**
  - The same restoration brings APPLY back in Qwen3-8B (+0.22) and Phi (+0.45), but not in Qwen3-4B.
  - Mistral and Gemma-2-9B have APPLY near floor.
- **Account.** The binding rewrites how rule entries are *addressed*. After integration, the key of a rule entry carries the identity of its codebook value, so a reader query formed from the decoded value can find that entry. The content returned does not depend on the binding.
  - This also explains the path effect of §5.3: without binding, the readers cannot locate the entry.
  - Application needs further steps, such as decoding the input key. That is why restoring the readers' keys is not always enough for it.

### 5.6 Single query heads (pre-registered G9)

*`t19`; Figure 9. Fresh bundles (seed 61), heads ranked on held-out bundles (seed 73).* In grouped-query models one key/value group serves 2–4 query heads, so the group-level result could be carried by the reader heads' group-mates. We therefore give clean keys at the rule tokens to selected **query heads** only; the other heads of the same groups keep the blocked keys. Giving clean keys to every query head of a group reproduces the group-level restoration exactly (tested).

| Model | blocked | readers' groups | top-1 / 2 / 4 / 8 / 16 reader heads only | group-mates of top-8 | random 8 |
|:---|---:|---:|:---|---:|---:|
| Qwen3-4B | 0.27 | 0.67 | 0.42 / 0.50 / 0.73 / 0.73 / 0.79 | 0.27 | 0.32 |
| Qwen3-8B | 0.15 | 0.77 | 0.46 / 0.67 / 0.69 / 0.73 / 0.73 | 0.19 | 0.27 |
| Mistral-7B | 0.31 | 0.54 | 0.33 / 0.42 / 0.44 / 0.48 / 0.44 | 0.33 | 0.31 |
| Gemma-2-9B | 0.56 | 0.90 | 0.58 / 0.67 / 0.77 / 0.90 / 0.92 | 0.50 | 0.51 |
| Llama-3.1-8B | 0.02 | 0.27 | 0.02 / 0.06 / 0.15 / 0.17 / 0.38 | 0.02 | 0.01 |

- **The reader heads carry the effect (P9b met, 4/5).** The top-8 reader heads alone recover 73–116 % of what their whole groups recover; Llama 58 %. Their group-mates recover nothing.
- **Few heads suffice (P9c).** Half of the group-level recovery takes one head in Qwen3-8B, two in Qwen3-4B, and four in Mistral, Gemma, Phi and Llama.
- **Margin over random heads (P9a not met, 3/5).** Top-8 minus random-8 is +0.38 to +0.46 in Qwen3-4B/8B and Gemma, and +0.15 to +0.17 (CIs excluding 0) in Mistral and Llama, below the pre-registered 0.20.
- Llama reads through more heads: its top-16 reader heads exceed the 8 groups.

### 5.7 Restoration window

*`t5`, `t11`; Figures 4–5.*

- Restoring clean later-source states after layer L brings rule reading back **only within a mid-layer window**: L ≈ 14–20 in Qwen3-4B/8B and L ≈ 12–18 in Phi. The window is the same for both formats and both orders.
- **Our account (post-hoc).** Integration is complete by the start of the window. The positions that come after the later source read it only after the window.

## 6 Why order matters only in JSON

*`t11`, `t11b`, `t16`; Figure 5.*

**Same heads.** The top-32 binding heads overlap strongly across formats and orders (Jaccard 0.49–0.73; random sets 0.02). Prose and JSON therefore do not use different heads.

**Same restoration window.** In codes with the rule first, restoration recovers close to that order's *clean* level (Qwen3-4B: 0.75 vs 0.79). The deficit lies in the clean computation, not in a missing mechanism.

**What differs is how much accuracy depends on the integration.** Measured post hoc as clean − blocked READ:
- in JSON it is asymmetric: −0.75 when the rule comes second versus −0.31 when the codebook comes second (Qwen3-4B);
- in prose it is nearly symmetric: −0.52 versus −0.43.

So in JSON, when the codebook comes second, the integration contributes less and the model is less accurate. In prose, every entry spells out its relation ("badge K is assigned to room V"; "room V moves to room W"), so the later source integrates the earlier one equally well in both orders. This matches our pre-registered hypothesis H-prose (P1a partly met, P1b met).

**A direct route in prose (pre-registered G7; fresh bundles; `t16`).** Rule reading does not logically need the codebook: the decoded value is already in the JSON prefix. In prose, the queried word ("Amber") also occurs verbatim in the rule entry ("room Amber moves to room Cedar"), so the entry can be matched directly.

- **Removing the codebook.** With the whole codebook blocked, reader heads keep part of their clean attention to the queried entry:

  | Setting | Attention kept |
  |:---|---:|
  | Qwen3-4B, prose | 25–34 % |
  | Qwen3-8B, prose | 50–76 % |
  | Gemma-2-9B, JSON | 46 % |
  | Other models, JSON | 5–19 % |

- **Partial blocks.** A partial block cuts off only the codebook value tokens, or only the top-32 binding heads.
  - In prose it is *worse* than removing the codebook (top-32 − whole: −0.08 to −0.46; 3 of 4 CIs exclude 0).
  - In JSON it never is (−0.02 to +0.35).
  - The partial block leaves the rule tokens with a corrupted binding that misleads the readers: their attention to the right entry also drops (−0.02 to −0.07).
- **The round-4 puzzle.** In round 4, Qwen3-8B's prose reading hardly depended on the integration. It is the strongest case of a direct route that runs throughout prose, not a separate mechanism. P7a–c were partly met: the effect is a property of the format rather than of the 8B model.

## 7 Adaptation (pre-registered; Qwen3-4B and Phi; 5 seeds)

*`t4`, `t4b`; Figure 3.*

**Setup.** Parameter-matched adapters on frozen backbones, trained for 256 steps on codebook-first items:
- QUERY;
- SEPARABLE, φ(q)⊙ψ(k);
- VECTOR_PAIR, 1+tanh(Qq+Kk) applied to each value before aggregation;
- LoRA on q/k/v/o, or on v only.

**Results.**
- **No transfer across order.**
  - Adapters trained on codebook-first items only reach 0.90–0.96 in distribution, but 0.43–0.63 with the rule first (frozen: 0.33).
  - Training on both orders reaches 0.70–0.94.
  - In Phi, rule-first APPLY stays ≤ 0.14.
- **The architectural hypothesis is negative.**
  - VECTOR_PAIR never beat the best control at composition.
  - It did not reduce Phi's leakage more than the others. Codebook-first leakage goes from 18.8 % frozen to 7.9–11.7 % after adaptation, with LoRA lowest.
- **Leakage induced by adaptation.**
  - Under aggressive training (lr 2e-3, or training on both orders), LoRA-qkvo leaks more than Separable and Vector pair, by 0.07–0.10 with CIs excluding 0.
  - Under mild training there are no differences.
  - With a single layer, Separable leaks least, and the pairwise component never helps.

## 8 Discussion and limitations

**What generalizes:**
- the reading/application dissociation (13 of 14 models, 7 families; the exception cannot read the rule);
- the need for later → earlier integration through shared value tokens;
- a two-stage binding → reader circuit (6 models, 5 families), whose interface is the keys at the rule tokens as read by a few reader query heads (two independent bundle sets);
- the JSON-specific order asymmetry (14 models, 7 families, up to 14B).

**What does not:**
- the order asymmetry, in prose (3 templates);
- a strong dependence on the integration, in prose, where a direct-match route exists;
- clean read/apply contrasts, in weak prose readers.

**Limits of the mechanism evidence:**
- 32 heads are not enough for binding, except in Gemma-2-9B. Beyond the strongest heads the binding is distributed, and recovery through direct paths is not monotonic in the number of heads.
- The sender decomposition depends on a pre-registered layer L*; for Llama-3.1-8B it fell outside the restoration window.
- Restoring the readers' keys brings back reading everywhere, but application in only 2–3 of 6 models.
- The restoration window is interpreted post hoc.
- Llama-3 was run from community mirrors of the official weights.

**Practical takeaways:**
- In structured prompts, put the lookup table before the rule.
- Prose descriptions of rules are robust to order.
- Fine-tuning on one layout does not transfer to another.
- When scope must be preserved under aggressive adaptation, prefer value-branch adapters to LoRA.

## 9 Reproducibility

- **Code release** (`README.md`). It contains:
  - the task generator and probes;
  - knockouts (per layer and per head), restoration, and binding and reader head analysis;
  - sender decomposition, receiver key/value restoration and per-query-head key restoration (`mech_path`);
  - the adapters;
  - a parallel single-GPU job runner with a VRAM budget and retries of failed stages;
  - a full reproduction spec (`tools/make_job_specs.py`, 131 stages);
  - `paper/analysis.py`.
- **Raw receipts** are in a separate archive, identified by its SHA256.
- **Compute:** about 2.5 T4 GPU-hours plus about 49 Colab compute units on an RTX PRO 6000.
