# Binding Makes Rules Addressable: Key-Mediated Retrieval of In-Context Rules

*Manuscript v5 | 17 November 2026 | Existing experimental data only*

## Abstract

A language model can retrieve an in-context rule without reliably composing it with a preceding lookup. We study this distinction with a synthetic codebook-and-rule task in 13 frozen instruction-tuned models spanning seven model families. Candidate log-probability scoring separates answer selection from free-generation schema failures. In structured prompts, all 13 models have higher point estimates for rule reading than for two-hop application, although the paired 95% interval excludes zero in 11; all 13 favor codebook-first over rule-first presentation for reading. Attention interventions identify a binding-to-reader pathway in six models. After blocking rule-to-codebook attention throughout the network, restoring clean keys at the rule positions preferentially recovers reading relative to restoring values. This key advantage replicates with intervals excluding zero in five of six models on fresh bundles; the remaining model has a positive estimate with a lower bound at zero. Intervening separately on query heads in grouped-query attention assigns most of the group-level recovery to the selected reader heads in four of five grouped-query models. Application is recovered less consistently. Three prose templates substantially alter the order effect and expose residual retrieval under complete integration blocks, with model-specific exceptions. Parameter-matched pairwise value-gating adapters provide no consistent advantage over simpler controls. Together, these results support key-mediated addressability as an important component of rule retrieval while identifying limits to its sufficiency for composition and selective rule use.

## 1 Introduction

Rules supplied in a prompt can serve several purposes. A model may be asked to report a rule's consequence for a given value, apply the rule after first retrieving that value, or answer a request that does not authorize the rule. These behaviors place different demands on the network. Reporting a rule consequence is a lookup when its input is supplied; applying it requires composition; withholding it requires interpreting scope. Treating all three as a single measure of instruction following can hide their different failure modes.

Consider a codebook containing `badge 317 -> Amber` and a procedure containing `Amber -> Cedar`. Given Amber, reading the procedure requires returning Cedar. Applying the procedure to badge 317 requires first retrieving Amber and then returning Cedar. An out-of-scope request requires retaining Amber. The rule may be present and readable even when the latter two decisions fail.

We use this controlled setting to ask how a rule becomes addressable by the attention heads that retrieve it. Our intervention sequence moves from source-to-source attention blocks, through head-level necessity and restricted sufficiency, to restoration of keys and values at the reader interface. The strongest evidence concerns one-hop rule retrieval. The application results are informative precisely because recovery of that interface does not consistently recover the complete two-hop computation.

Our contributions are threefold. First, we characterize reading, application, source order and scope in 13 instruction-tuned models, and report paired uncertainty rather than treating every positive difference as a successful replication. Second, in six models from five families, we combine a binding-to-reader attention-path effect with key/value restoration and fresh-bundle replication. Per-query-head interventions distinguish a reader from the other heads sharing its key/value group. Third, we show that prose changes the behavioral dependence on source order and integration, while a five-seed adapter study finds no consistent benefit from the proposed pairwise value gate.

This is a controlled study of a retrieval interface in pretrained models. It does not claim that binding through keys, writer-reader organization, or pairwise message gating is new in itself. Its contribution lies in connecting these mechanisms to explicitly separated reading, composition and scope decisions, and in testing the receiver interface under a common integration block.

## 2 Related work

**Composition and source order.** The compositionality gap measures the difference between answering constituent questions and their composition [1]. Guo et al. study in-context two-hop reasoning with distractors and analyze learned algorithms in three-layer transformers [2]. Their natural-language dataset permutes premises, while their symbolic mechanistic setting keeps the two premises of each chain in order. We therefore do not characterize that work as uniformly fixing premise order. Other work studies latent multi-hop reasoning and delayed second-hop processing [3, 4]. Premise permutation, misordered context and the causal attention mask also produce order sensitivity [5-7]. We add a controlled comparison of structured and prose encodings, with interventions on integration between the two sources.

**Binding and retrieval.** Feng and Steinhardt identify binding-ID representations [8]. Gur-Arieh et al. show that positional, lexical and reflexive mechanisms can jointly support retrieval of bound entities [9]. Induction heads and retrieval heads provide related accounts of content-dependent copying and retrieval [10, 11]. We use the operational name *reader* for heads ranked by attention to the queried rule entry; we do not propose a new universal class of head. Our binding-head score measures duplicate-token attention across the two source spans, rather than directly estimating a binding-ID subspace.

**Keys as an interface.** Oh and Demberg report key-dominated binding in Llama and query/key binding signatures in Gemma on dynamic entity tracking [12]. Wang et al. causally separate QK routing from Value updates in function vectors and find more consistent contributions from routing [13]. Gao and Gao identify writers, readers and key/value memories at label-token positions in few-shot classification, including restricted key and value interventions [14]. Our experiments complement these accounts with rule-token restoration under a source-integration block and with a per-query-head intervention in grouped-query models. The tasks and corruptions differ, so differences in recovery should not be read as contradictions between studies.

**Adaptation.** LoRA provides a low-rank adaptation baseline [15]. Feature-wise gates on pair-dependent messages have precedents in gated graph networks and self-attention [16, 17]. We accordingly treat pairwise value gating as an empirical hypothesis about selective rule use, not as a claim to a new operator. Its failure to consistently outperform parameter-matched controls is part of the result.

## 3 Experimental design

### 3.1 Task and experimental factors

A bundle contains a six-entry codebook C, a derangement P of its values, and an alternative derangement Q. The optional distractor makes Q visible. Codebooks, mappings and query assignments are generated deterministically from the data seed, split, bundle index and template. Test-ID and test-OOD splits differ in request-cue families; this is cue generalization within a synthetic task, not an evaluation on natural-world domains.

Each mode uses a distinct input key within a bundle. This prevents reuse of another mode's answer but means a READ/APPLY contrast is matched by bundle and condition, not by an identical queried key. READ_DEC requests C(x). READ_RULE requests P(v) with v = C(x) supplied in the teacher-forced answer prefix. APPLY_P requests P(C(x)); APPLY_Q requests Q(C(x)) when Q is present. NONAPPLY requests C(x) without applying a procedure. We use *scope leakage* for choosing the P-transformed value in NONAPPLY.

The main frozen probe crosses two framings, neutral and default-procedure, with distractor absent/present and codebook-first/rule-first order. Each mode and order typically has 192 observations: 48 bundles across the two test splits, with four framing-by-distractor settings per bundle. A separate framing control adds a content-free filler sentence. Mechanistic experiments generally use neutral framing, no distractor and 48 bundles; the corresponding condition is stated with each result.

The structured format renders codebook and rule maps as JSON over nonce symbols such as x17 and z42. Three prose templates describe badges and rooms, flights and gates, or inventory items and shelves. These templates change vocabulary, syntax and relational wording together. All retain the JSON answer contract. Thus this comparison tests the complete encodings, not a punctuation-only intervention.

### 3.2 Models and scoring

We evaluate 13 models: Qwen3 at 1.7B, 4B, 8B and 14B; Qwen2.5-3B; Phi-3.5-mini; OLMo-2-7B; Granite-3.3-8B; Gemma-2 at 2B and 9B; Mistral-7B; Llama-3.1-8B; and Llama-3.2-3B. Counting Qwen2.5 and Qwen3 together yields seven families. Not every model has every mechanistic or prose experiment; Appendix A records coverage.

Four distinct candidate values are scored by summed continuation log-probability. Scoring starts after the longest token prefix shared by all candidate sequences, avoiding an arbitrary boundary preference. READ_RULE's prefix includes the correct decoded value; APPLY's prefix contains only the answer field. This intentionally separates one-hop reading from composition. It is not evidence that the model independently generated the supplied intermediate value. Chance accuracy among the four candidates is 0.25, and restricted-candidate accuracy is not unrestricted generation accuracy.

For Gemma, which rejects a system turn, the contract opens the user turn. Llama checkpoints are obtained through the unsloth mirrors recorded in the manifests; no independent weight-equivalence test is claimed here. The date in Llama-3.2's template is pinned. The exact model identifiers and revisions remain in the archived manifests.

### 3.3 Interventions and head selection

An integration block sets selected attention logits to negative infinity from the later source tokens to the earlier source span. In codebook-first experiments, rule tokens are blocked from reading the codebook. A sham blocks the same number of contract tokens; it controls span size but does not make the removed content semantically interchangeable. Head blocks restrict this operation to selected layer/head pairs. Random-head baselines provide size-matched controls, with three draws where specified.

Binding heads are ranked by duplicate-token attention across source spans. Reader heads are ranked by attention from the answer position to the queried rule entry. The path diagnostic records reader attention to that entry minus the mean attention to other entries; it is an attention contrast, not an unadjusted probability mass. Round-5 and round-6 restoration experiments rank heads on held-out bundles, separate from the evaluation bundles. Earlier localization runs are exploratory and do not all have this separation.

Receiver restoration starts from the fully blocked computation. Selected key/value groups receive clean K, clean V, or both at rule-token positions. The group set is induced by the top-eight reader query heads and can contain fewer than eight groups. The control restores the groups induced by eight random heads. These are clean-state interventions for causal diagnosis, not a deployable inference method.

For per-query-head restoration, post-RoPE keys are expanded to the query-head layout inside the attention kernel. Only selected heads receive the clean keys; their group-mates retain blocked keys. This intervenes on a particular head's K-input path, not on an independently learned K projection. The shared KV cache is unchanged. CPU tests verify that restoring every query head of a group equals restoring that group's keys.

### 3.4 Statistical analysis and evidence status

We report accuracy differences with 95% percentile intervals from 4,000 paired cluster-bootstrap resamples. Matching preserves every fixed experimental factor. READ/APPLY comparisons match bundle, split, framing, distractor, control and source order; order comparisons retain mode and change only order. Bundles are resampled as clusters so repeated conditions within a bundle are not treated as independent. Mechanistic contrasts match bundle, split and mode within their single condition; random draws are averaged per item before comparison.

Adapter leakage intervals follow the recorded seed-by-bundle clustering scheme. They describe the five observed training seeds and the evaluation bundles; they are not a full hierarchical uncertainty estimate over arbitrary future training runs. All intervals are pointwise, with no family-wise multiplicity correction. Intervals including zero are not equivalence tests.

Local experiment plans in docs/ record predictions and intended criteria. We distinguish their outcomes from exploratory analyses and do not claim an externally timestamped registration. This manuscript corrects a prior analysis implementation that omitted framing from some matching keys. No inference or training was rerun for this revision; the correction, combined-model summaries and stricter audit of the planned criteria use existing receipts only.

## 4 Reading, composition and scope

### 4.1 Structured prompts expose a reading/application gap

Table 1 summarizes the main frozen probe. Reading exceeds application in all 13 point estimates. Paired intervals exclude zero in 11 models. Phi-3.5 has a positive difference of 0.146 with interval [-0.010, 0.297]; Llama-3.2-3B has a small difference of 0.021 [-0.099, 0.146] and reads below chance. These two cases should not be counted as statistically resolved reading/application dissociations. The result is nevertheless broad across the other models: for example, Qwen3-4B reads at 0.95 and applies at 0.64, while Gemma-2-9B reads at 0.92 and applies at 0.29.

**Table 1. Frozen structured-task accuracy and observed scope leakage.** bf/rf denote codebook-first/rule-first. Leak counts are for bf NONAPPLY. Each displayed accuracy pools 192 items over 48 bundle clusters. APPLY denotes APPLY_P. Values come from the frozen tables; condition-matched intervals for all 13 models are in t20_all_models.

| Model | READ bf | READ rf | APPLY bf | APPLY rf | Leak bf |
|:---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 0.87 | 0.48 | 0.19 | 0.04 | 7/192 |
| Llama-3.2-3B | 0.17 | 0.01 | 0.15 | 0.02 | 8/192 |
| Gemma-2-2B | 0.71 | 0.10 | 0.06 | 0.01 | 5/192 |
| Qwen2.5-3B | 0.78 | 0.32 | 0.12 | 0.08 | 0/192 |
| Phi-3.5-mini | 0.80 | 0.38 | 0.66 | 0.02 | 29/192 |
| Qwen3-4B | 0.95 | 0.73 | 0.64 | 0.36 | 0/192 |
| Mistral-7B | 0.65 | 0.45 | 0.10 | 0.13 | 0/192 |
| OLMo-2-7B | 0.71 | 0.32 | 0.12 | 0.10 | 1/192 |
| Qwen3-8B | 0.95 | 0.76 | 0.47 | 0.26 | 1/192 |
| Granite-3.3-8B | 0.70 | 0.47 | 0.27 | 0.13 | 6/192 |
| Llama-3.1-8B | 0.73 | 0.35 | 0.57 | 0.22 | 17/192 |
| Gemma-2-9B | 0.92 | 0.71 | 0.29 | 0.15 | 0/192 |
| Qwen3-14B | 0.95 | 0.61 | 0.81 | 0.49 | 0/192 |

All 13 have a positive codebook-first advantage for reading, with paired intervals excluding zero. The differences range from 0.167 in Llama-3.2-3B to 0.609 in Gemma-2-2B. This is a consistent directional effect in this structured task, including the weak reader. Many failed applications select the untransformed codebook value: the pooled stop-at-base rate is 0.84 for Qwen3-1.7B and 0.96 for Gemma-2-2B. This rate is not universal; it is 0.16 in Qwen3-14B.

### 4.2 Scope and free-generation checks

Scope leakage varies by model and framing. In the main codebook-first probe, Phi leaks on 29/192 items and Llama-3.1-8B on 17/192. Qwen3-4B, Qwen3-14B and Gemma-2-9B have zero observed leakage in the structured frozen probe; these finite-sample observations are not guarantees of perfect scope control. Prose can increase leakage, as in OLMo's room template, with 63/384 items across orders, and Mistral's 23/192 codebook-first items.

On fresh Phi bundles, a default-procedure description reduces leakage from 31/96 under neutral framing to 17/96; a content-free sentence in the same slot gives 30/96. The default-minus-filler difference is -0.14 [-0.22, -0.05]. Thus the reduction is not explained simply by inserting text of similar length. Because APPLY remains about 0.67 for neutral and default framing, low leakage is not solely an application-floor artifact in this control.

The free-generation audit covers four models. Between 26% and 67% of sampled generations fail to map to the required candidate schema. Among valid parses, agreement with the log-probability choice is 90-100%. Some APPLY cells have no valid parses, so this conditional agreement does not establish unrestricted application competence. We use candidate scores to measure answer preference while reporting schema failure separately.

## 5 A binding-to-reader pathway

### 5.1 Integration blocks and head-level evidence

Blocking rule-to-codebook attention reduces reading relative to the size-matched sham in 12 of 13 models, counting the weak Llama-3.2 reader descriptively. Mistral is the exception: block-minus-sham is -0.15 [-0.31, 0.02], and the sham itself reduces reading by about 0.16. The result supports dependence on the integration path in most structured-task models, with an important limitation to the sham's specificity in Mistral.

Narrower Qwen3-4B interventions implicate shared values. Blocking codebook-value tokens lowers READ by 0.54 relative to clean, whereas blocking codebook-key tokens lowers it by only 0.02 with an interval including zero. The whole-codebook block lowers READ by 0.75. Value-token blocking is therefore strongly damaging but need not equal whole-span blocking on every endpoint. Blocking layers 24-35 changes READ by only -0.02 [-0.06, 0.00], whereas earlier bands are damaging.

Blocking the top-eight binding heads reduces Qwen3-4B APPLY to 0.04 versus 0.61 under random-eight blocking, and READ to 0.56 versus 0.94. Phi shows corresponding top-minus-random effects of -0.51 on APPLY and -0.23 on READ. Restricting integration to the top-32 binding heads restores part of reading in Qwen3-4B/8B and Phi, but leaves substantial deficits. Llama-3.1-8B recovers particularly little: READ is 0.12 with only these heads, against 0.65 clean. Compact-head sufficiency for the full behavior is not established.

Across six models, binding-head blocking also reduces the reader attention contrast to the queried rule entry. For Qwen3-4B it falls from 0.416 to 0.107; for Phi, from 0.526 to 0.184; for Llama-3.1-8B, from 0.280 to 0.031. Top-minus-random differences are negative with intervals excluding zero in all six. Blocking reader heads also reduces application relative to random-head blocks in models with sufficient clean application performance. These interventions link source integration to downstream retrieval but do not identify every step of composition.

![Figure 1. Intervention logic for the binding-to-reader pathway. The proposed information-flow account is supported by source blocks, reader-attention changes and restricted key restoration. The additional computation required for APPLY is not fully identified.](figures/f10_circuit.png)

### 5.2 Receiver restoration identifies a key-mediated interface

Table 2 reports the original receiver experiment in five models. The rule-to-codebook block remains active in every layer. Restoring clean K+V in the groups of the top-eight readers improves READ over restoring random-head groups by 0.16-0.45, with intervals excluding zero. Restoring only K yields substantially more recovery than restoring only V. For Qwen3-4B, reading rises from 0.19 blocked to 0.75 with K, against 0.21 with V. For Phi, K reaches 0.81 versus 0.46 with V.

**Table 2. READ accuracy under receiver restoration, original bundles.** Each variant has 48 bundle clusters. Reader-group K and V are restored only at rule-token positions. Random restores K+V in the groups induced by eight random heads, averaged over three draws. The last column is the paired K-only minus V-only contrast.

| Model | Clean | Blocked | K+V | K only | V only | Random | K - V [95% CI] |
|:---|---:|---:|---:|---:|---:|---:|:---|
| Qwen3-4B | 0.96 | 0.19 | 0.77 | 0.75 | 0.21 | 0.33 | +0.54 [0.40, 0.69] |
| Qwen3-8B | 1.00 | 0.21 | 0.79 | 0.71 | 0.25 | 0.34 | +0.46 [0.31, 0.60] |
| Phi-3.5-mini | 0.92 | 0.52 | 0.77 | 0.81 | 0.46 | 0.53 | +0.35 [0.19, 0.50] |
| Mistral-7B | 0.60 | 0.17 | 0.46 | 0.38 | 0.15 | 0.22 | +0.23 [0.10, 0.35] |
| Gemma-2-9B | 0.94 | 0.69 | 0.88 | 0.88 | 0.71 | 0.72 | +0.17 [0.06, 0.27] |

The result replicates on new evaluation bundles with a new held-out head-selection seed, adding Llama-3.1-8B (Figure 2). Reader-group K+V beats random-group K+V by 0.26-0.54 in all six codes models, with intervals excluding zero. K-only exceeds V-only in all six point estimates; the intervals exclude zero in five, while Mistral's lower bound is zero. This meets the planned replication criterion of five out of six. Fresh-bundle replication tests data stability with the same checkpoints and implementation, not independence of models or laboratories.

Values-only restoration gives little reading recovery in the original codes experiments. Similar K-only and K+V accuracies support a predominantly key-mediated effect under this corruption, but do not prove values are irrelevant to ordinary retrieval. Both are necessary components of attention, and the clean-versus-blocked value difference can be small even when the value path carries essential content.

![Figure 2. Receiver restoration on fresh bundles. Bars report READ accuracy for blocked, K-only, V-only and clean conditions in six codes models; each condition has 48 bundles. Paired K-minus-V intervals, rather than separate bar intervals, provide the inferential comparison.](figures/f11_receiver_fresh.png)

The distinction between retrieval and application remains essential. In the original five-model receiver experiment, K+V minus random-group restoration on APPLY exceeds the planned 0.20 threshold in Qwen3-8B (+0.22) and Phi (+0.45), but not in the other three. The planned requirement of three models is not met. Fresh Llama bundles give a smaller positive application effect of +0.09 [0.02, 0.17]. Thus restoring this interface can facilitate composition, but it does not suffice consistently for the complete computation.

### 5.3 Individual reader query heads carry much of the recovery

Group-level restoration can benefit multiple query heads sharing K/V projections. We therefore restore clean keys separately for selected query heads while keeping the other heads of their groups blocked. Phi has one query head per KV group and is not included in the five-model GQA comparison.

**Table 3. Per-query-head key restoration on fresh codes bundles.** Group K restores keys for the groups induced by the top-eight readers. Reader K restores keys only along those eight query-head paths; mates restores only their remaining group-mates. Recovery fraction is (Reader K - blocked)/(Group K - blocked), computed from unrounded accuracy. Each variant has 48 bundles.

| Model | Blocked | Group K | Reader K | Mates K | Random 8 | Recovery fraction |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.27 | 0.67 | 0.73 | 0.27 | 0.32 | 1.16 |
| Qwen3-8B | 0.15 | 0.77 | 0.73 | 0.19 | 0.27 | 0.93 |
| Mistral-7B | 0.31 | 0.54 | 0.48 | 0.33 | 0.31 | 0.73 |
| Gemma-2-9B | 0.56 | 0.90 | 0.90 | 0.50 | 0.51 | 1.00 |
| Llama-3.1-8B | 0.02 | 0.27 | 0.17 | 0.02 | 0.01 | 0.58 |

The readers alone recover at least 70% of the group effect in four of five models, satisfying the planned three-of-five criterion. Their group-mates produce much smaller changes. In Qwen3-8B, one reader takes READ from 0.15 to 0.46, half of the group-level recovery. Llama is more distributed: the top-eight readers recover 58% of the group effect, and the top-16 exceed the original eight groups' recovery.

Top-eight minus random-eight is positive with intervals excluding zero in all five models. However, the stronger planned margin of at least 0.20 in four models is not met: only Qwen3-4B, Qwen3-8B and Gemma pass it. Mistral and Llama reach approximately +0.17 and +0.15. This distinction between a positive effect and the specified success threshold is retained.

### 5.4 What the interventions establish

The joint evidence supports an addressability account: integration changes rule-position keys in a way that lets downstream readers locate the queried entry. This is an interpretation of intervention outcomes, not a direct semantic decoding of all information in K. Restoring K partially repairs the retrieval interface even when source integration remains blocked. It does not establish a complete minimal circuit or show that a deployed model can obtain clean keys without performing the computation being restored.

Restoring clean rule-token residual states at selected middle layers provides complementary evidence. Recovery is strongest around the sampled mid-layer region, with model-specific widths and incomplete restoration. The interpretation that integration has completed before downstream readers use it is post hoc. We do not generalize a single universal restoration window to every family.

The sender direct-contribution decomposition has a stricter limitation. Its local plan required restoring all components at the chosen layer to bring READ within 0.10 of clean before interpreting attention-versus-MLP differences. Unrounded deficits are 0.0625 for Qwen3-4B, 0.1250 for Qwen3-8B, 0.1458 for Phi and 0.1042 for Mistral. Only Qwen3-4B passes. Llama's later extension also fails substantially. We therefore treat the multi-model sender decomposition as exploratory, rather than a successful confirmatory demonstration that attention alone writes the binding. Appendix B reports these outcomes, including non-monotonic partial patches.

## 6 Format changes the dependence on order and integration

The structured-task codebook-first advantage does not carry over uniformly to prose. In the room template, Qwen3-4B reads at 0.63 with the book first and 0.74 with the rule first; Gemma-2-9B reads at 0.83 and 0.88. The gates and shelves templates also show smaller or reversed differences. These results motivate a format-dependent claim rather than a claim that order never matters in prose.

For the gates template, Phi has a reversed gap of -0.188 [-0.302, -0.068]. In shelves, Qwen3-8B also has a reversed gap of -0.094 [-0.167, -0.021], while OLMo has a positive gap of +0.083 [0.005, 0.167] from low absolute reading accuracy. The local plan's descriptive rule, absolute gap below 0.10 or an interval containing zero, can be satisfied even by a statistically nonzero small effect; it is not an equivalence test. Low-reading models also cannot provide strong evidence of order robustness. Full condition-matched results are in t21_format_order.

Binding-head sets overlap across codes/prose and order in Qwen3-4B/8B: top-32 Jaccard similarity is 0.49-0.73, against about 0.02 for random sets. This supports substantial reuse of heads, not identity of every head or the entire algorithm. Prose need not invoke a wholly separate circuit to reduce the order gap.

A direct-match route is a plausible contributor. READ_RULE already supplies the decoded value in its answer prefix, and a prose rule repeats that value as a word. Under whole-codebook integration blocks, the reader attention contrast retains 25-34% of clean in Qwen3-4B prose and 50-76% in Qwen3-8B prose. The corresponding structured experiments retain 5-19% in most tested models; Gemma is an exception at 46%, and Llama is lower at about 3%. These are ratios of the entry-minus-other-entries attention diagnostic.

Partial blocks can be more harmful than whole-codebook blocks. Across four Qwen prose settings, top-32 binding-head blocking minus whole blocking changes READ by -0.08 to -0.46, with three intervals excluding zero. In the structured settings, the same contrast is not significantly negative. One account is that a partial integration leaves misleading representations that interfere with an alternate retrieval route. Existing interventions establish the non-monotonic behavioral pattern; they do not uniquely distinguish corrupted binding from compensation or other nonlinear interactions. The narrower planned claim that the effect is specific to Qwen3-8B is not supported across all settings.

## 7 Adaptation: useful fitting without a pairwise-gate advantage

Adapters are attached to frozen backbones and trained for 256 steps. The central Qwen3-4B comparison uses four selected layers, parameter-matched Separable, Vector Pair and LoRA-qkvo adapters, and five training seeds. Auxiliary settings include query-only and LoRA-v controls. Vector Pair gates each value before aggregation using a feature-wise factor `1 + tanh(Aq_i + Bk_j)`; the comparison tests whether this pair dependence improves composition or scope control.

Training only on codebook-first items improves in-distribution application, but cross-order performance remains lower and depends strongly on method and seed. At learning rate 5e-4, Separable, Vector Pair and LoRA achieve bf accuracies of 0.92, 0.90 and 0.96, versus rf accuracies of 0.48, 0.43 and 0.63. These results show incomplete generalization across order, not absence of transfer. Training on both orders reaches rf accuracies of 0.80, 0.70 and 0.94 respectively. Phi's adapted rf application remains at or below 0.14 in the reported setting.

**Table 4. Selected five-seed Qwen3-4B adaptation results.** Evaluation uses neutral framing, both distractor settings and both orders. Leakage pools 480 NONAPPLY items per method and setting. The full comparison, including auxiliary models and learning rates, is in t4_adaptation.

| Training | Method | APPLY bf | APPLY rf | Leak count |
|:---|:---|---:|---:|---:|
| bf, lr 5e-4 | Separable | 0.92 | 0.48 | 8/480 |
| bf, lr 5e-4 | Vector Pair | 0.90 | 0.43 | 6/480 |
| bf, lr 5e-4 | LoRA-qkvo | 0.96 | 0.63 | 9/480 |
| bf, lr 2e-3 | Separable | 0.95 | 0.68 | 23/480 |
| bf, lr 2e-3 | Vector Pair | 1.00 | 0.52 | 14/480 |
| bf, lr 2e-3 | LoRA-qkvo | 0.83 | 0.29 | 58/480 |
| bf+rf, lr 5e-4 | Separable | 0.90 | 0.80 | 7/480 |
| bf+rf, lr 5e-4 | Vector Pair | 0.83 | 0.70 | 4/480 |
| bf+rf, lr 5e-4 | LoRA-qkvo | 0.95 | 0.94 | 54/480 |

Vector Pair provides no consistent advantage over the simpler controls. Its strong bf accuracy at high learning rate does not translate to the best rf accuracy or a consistently superior accuracy/leakage trade-off. Under aggressive Qwen training, LoRA-qkvo leaks more than Separable or Vector Pair: differences are approximately 0.07-0.10, with the recorded seed-by-bundle intervals excluding zero. Mild training does not show the same separation. Single-layer results also do not favor the pairwise component. These are setting-specific observations; they do not establish that value-branch adapters generally preserve scope better than LoRA across tasks.

## 8 Discussion and limitations

The principal positive finding is a causal interface for retrieving an in-context rule. Integration-block, head-path and receiver-restoration evidence converge on rule-position keys as an important addressability channel. The result extends existing binding and writer-reader accounts to a task that separates reading, two-hop application and withholding a rule. We avoid claiming the first discovery of key-based binding, a complete explanation of application, or a universally minimal head set.

Several limits constrain the interpretation. First, the task has six entries, synthetic mappings and restricted candidates. The three prose templates remain synthetic and retain structured answers. They do not establish performance on realistic policies, long documents or open-ended instruction following. Second, READ_RULE supplies the correct intermediate value and uses a different input key from APPLY; the observed contrast is designed to separate operations, but it also changes the available information. Third, behavior and mechanism are not uniformly strong across families: weak readers limit contrasts, Mistral's sham is damaging, and Llama requires more distributed restoration.

Fourth, clean activation patches can create hybrid states outside the model's ordinary trajectory. Their positive effects identify causal sensitivity under the specified block, not a naturally isolated information channel or an implementable repair without an oracle. The non-monotonic partial patches and the failed sender sanity checks make this limitation concrete. Fifth, K-only versus V-only results concern differences between clean and blocked activations; small V-restoration effects do not imply that values carry no task content.

Sixth, model checkpoints, bundles and templates are a finite selected set. Head counts and restoration layers were not exhaustively optimized for every family. The five-seed adaptation intervals use the recorded clustering convention rather than a complete hierarchy over training runs. Multiple reported intervals are pointwise and exploratory comparisons are not multiplicity-adjusted. The local plans improve transparency but are not an externally timestamped registration.

Finally, source-order effects are encoding dependent. The current data support presenting the codebook before the rule for the tested JSON task; they do not support a universal instruction to use prose, since prose can reduce absolute reading accuracy or increase leakage. The practical lesson is to evaluate retrieval, composition and scope separately when changing a prompt representation or adaptation method.

## 9 Reproducibility and conclusion

All reported experiments precede this manuscript revision. The task generator, probes, interventions, adapters and run plans are in the project. paper/analysis.py rebuilds the result tables and figures from the original layout. The reproduction job contains 131 stages for users with suitable GPU resources, but no such stages were executed to produce this revision. Existing checkpoint identities, configurations and source hashes are retained in the receipts and manifests.

The original-data archive contains 21,272 outer entries and has SHA256 `55fd2767737cf8143f529ce24a4f0e3c409dcc64615e84fe4e964be6fa44c748`. It is available locally with this project; public access is not claimed. Its README currently describes private distribution. This availability limit should accompany any release until a public archive exists. CPU reanalysis verifies receipt hashes and reconstructs the existing results; software smoke tests use tiny random models and are distinct from pretrained-model evidence.

In this controlled setting, making a rule addressable is an important step toward using it. Binding-dependent changes in rule-token keys help reader heads retrieve the appropriate entry, and restricted restoration can recover that reading under a continuing integration block. The remaining composition deficits, format-specific routes and negative adapter comparison show why retrieval competence should be measured separately from successful and appropriately scoped application.

## References

1. Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., and Lewis, M. (2023). Measuring and Narrowing the Compositionality Gap in Language Models. Findings of EMNLP. [arXiv:2210.03350](https://arxiv.org/abs/2210.03350).

2. Guo, T., Zhu, H., Zhang, R., Jiao, J., Mei, S., Jordan, M. I., and Russell, S. (2025). How Do LLMs Perform Two-Hop Reasoning in Context? [arXiv:2502.13913](https://arxiv.org/abs/2502.13913).

3. Yang, S., Gribovskaya, E., Kassner, N., Geva, M., and Riedel, S. (2024). Do Large Language Models Latently Perform Multi-Hop Reasoning? ACL. [arXiv:2402.16837](https://arxiv.org/abs/2402.16837).

4. Biran, E., Gottesman, D., Yang, S., Geva, M., and Globerson, A. (2024). Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries. EMNLP. [arXiv:2406.12775](https://arxiv.org/abs/2406.12775).

5. Chen, X., Chi, R. A., Wang, X., and Zhou, D. (2024). Premise Order Matters in Reasoning with Large Language Models. ICML. [arXiv:2402.08939](https://arxiv.org/abs/2402.08939).

6. Yu, S., Kim, I., Song, J., Lee, S., Park, J., and Yoon, S. (2025). Unleashing Multi-Hop Reasoning Potential in Large Language Models through Repetition of Misordered Context. Findings of NAACL. [arXiv:2410.07103](https://arxiv.org/abs/2410.07103).

7. Ok, H., and Lee, J. (2026). Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models. [arXiv:2601.14152](https://arxiv.org/abs/2601.14152).

8. Feng, J., and Steinhardt, J. (2023). How do Language Models Bind Entities in Context? [arXiv:2310.17191](https://arxiv.org/abs/2310.17191).

9. Gur-Arieh, Y., Geva, M., and Geiger, A. (2025). Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context. [arXiv:2510.06182](https://arxiv.org/abs/2510.06182).

10. Olsson, C., Elhage, N., Nanda, N., et al. (2022). In-context Learning and Induction Heads. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895).

11. Wu, W., Wang, Y., Xiao, G., Peng, H., and Fu, Y. (2024). Retrieval Head Mechanistically Explains Long-Context Factuality. [arXiv:2404.15574](https://arxiv.org/abs/2404.15574).

12. Oh, S., and Demberg, V. (2026). A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models. [arXiv:2606.08644](https://arxiv.org/abs/2606.08644).

13. Wang, E., Wang, Y., Bakalova, A., and Hahn, M. (2026). How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning. [arXiv:2605.16591](https://arxiv.org/abs/2605.16591).

14. Gao, A., and Gao, W. (2026). A Circuit for In-Context Learning. [arXiv:2611.05344](https://arxiv.org/abs/2611.05344).

15. Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

16. Bresson, X., and Laurent, T. (2017). Residual Gated Graph ConvNets. [arXiv:1711.07553](https://arxiv.org/abs/1711.07553).

17. Zhao, H., Jia, J., and Koltun, V. (2020). Exploring Self-attention for Image Recognition. CVPR. [CVF proceedings](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Exploring_Self-Attention_for_Image_Recognition_CVPR_2020_paper.html).

## Appendix A. Experimental coverage and source inventory

All 13 models have structured frozen probes and broad integration knockouts. Six have the binding-to-reader circuit and receiver/head analyses: Qwen3-4B, Qwen3-8B, Phi-3.5-mini, Gemma-2-9B, Mistral-7B and Llama-3.1-8B. Five of these are grouped-query models; Phi is the one-head-per-group control. Sender decomposition excludes Gemma because its normalization architecture is outside the decomposition identity used here. Prose route interventions cover Qwen3-4B/8B in rooms and shelves, while the broader frozen prose coverage differs by model. The gates/shelves frozen probes cover five models each.

The main frozen probe uses data seed 17 unless the source configuration specifies a phase-specific seed. The room-template probes use seed 41 in the planned reproduction, and the later gates/shelves probes use seed 53. Receiver and sender evaluation uses seed 41 with selection seed 53; fresh receiver and per-query-head evaluation uses seed 61 with selection seed 73. These seed labels refer to the deterministic generator, not to independent model trainings. Exact configurations and checkpoint revisions in the archived stage manifests take precedence over shorthand model names.

**Table A1. Reproduction labels and checkpoint identifiers.** Revisions are stored in each stage's manifest rather than inferred from the mutable model repository's current main branch.

| Paper label | Checkpoint identifier |
|:---|:---|
| Qwen3-1.7B | Qwen/Qwen3-1.7B |
| Qwen3-4B | Qwen/Qwen3-4B-Instruct-2507 |
| Qwen3-8B | Qwen/Qwen3-8B |
| Qwen3-14B | Qwen/Qwen3-14B |
| Qwen2.5-3B | Qwen/Qwen2.5-3B-Instruct |
| Phi-3.5-mini | microsoft/Phi-3.5-mini-instruct |
| OLMo-2-7B | allenai/OLMo-2-1124-7B-Instruct |
| Granite-3.3-8B | ibm-granite/granite-3.3-8b-instruct |
| Gemma-2-2B | google/gemma-2-2b-it |
| Gemma-2-9B | google/gemma-2-9b-it |
| Mistral-7B | mistralai/Mistral-7B-Instruct-v0.3 |
| Llama-3.1-8B | unsloth/Llama-3.1-8B-Instruct |
| Llama-3.2-3B | unsloth/Llama-3.2-3B-Instruct |

The raw-to-table mapping is paper/layout_original.json. The table inventory is: t1/t7/t12/t17 for frozen codes; t2/t7b/t12b/t17b for broad knockouts; t3 for narrow knockouts; t4/t4b for adaptation; t5 for early restoration and binding heads; t6/t6b/t6c for generation and prose; t9 for framing; t10/t18 for circuit/path tests; t11/t11b for format restoration and head overlap; t13 for gates/shelves; t14 for sender decomposition; t15/t19b/t22 for receiver restoration; t16 for route tests; and t19 for query-head interventions. New summary tables t20 and t21 consolidate existing data with corrected matching.

**Table A2. Fresh-bundle receiver replication.** Each condition has 48 bundle clusters, evaluation seed 61 and held-out head-selection seed 73. Both intervals are paired READ accuracy differences. K+V minus random uses the reader-induced groups and the mean of three random-head-group draws. Mistral's K-minus-V lower bound is zero, so it is not counted among the five resolved K advantages.

| Model | K only | V only | K - V [95% CI] | K+V - random [95% CI] |
|:---|---:|---:|:---|:---|
| Qwen3-4B | 0.67 | 0.27 | +0.40 [0.27, 0.54] | +0.54 [0.40, 0.68] |
| Qwen3-8B | 0.77 | 0.21 | +0.56 [0.42, 0.71] | +0.54 [0.41, 0.67] |
| Phi-3.5-mini | 0.83 | 0.46 | +0.38 [0.25, 0.52] | +0.41 [0.28, 0.53] |
| Mistral-7B | 0.54 | 0.40 | +0.15 [0.00, 0.29] | +0.32 [0.18, 0.45] |
| Gemma-2-9B | 0.90 | 0.60 | +0.29 [0.15, 0.44] | +0.33 [0.20, 0.47] |
| Llama-3.1-8B | 0.27 | 0.00 | +0.27 [0.15, 0.40] | +0.26 [0.14, 0.39] |

## Appendix B. Planned criteria and departures

The following table reports key criteria in the local experiment plans using unrounded observations. Results are not relabeled as successes when they merely have the predicted sign.

| Criterion | Recorded requirement | Outcome in existing data |
|:---|:---|:---|
| Circuit P2c | Binding-head block lowers reader attention | Met in 6/6 tested models |
| Sender P5a | Full residual add-back within 0.10 of clean | Only Qwen3-4B passes among original four; pooled confirmatory G5 interpretation withheld |
| Sender P5c | Concentrated recovery and increasing dose | Not met; exploratory partial patches can be non-monotonic |
| Receiver P6a | APPLY advantage at least 0.20 in at least 3/5 | Not met; 2/5 meet the margin |
| Heads P9a | Reader-vs-random advantage at least 0.20 in at least 4/5 GQA models | Not met; 3/5 meet the margin |
| Heads P9b | At least 70% of group recovery in at least 3/5 | Met; 4/5 |
| Replication P10a | K-only exceeds V-only with CI excluding 0 in at least 5/6 | Met; 5/6 |
| Replication P10b | K+V-vs-random advantage at least 0.15 and CI excluding 0 in at least 5/6 | Met; 6/6 |

The sender full-add-back criterion tests behavioral restoration at the chosen layer, beyond the algebraic identity validated in unit tests. Later integration remains blocked, so a correct identity can still fail this criterion. Attention-minus-MLP and non-monotonic top-k results remain in t14 as exploratory analyses; they do not establish multi-family confirmatory evidence for an attention-exclusive sender.

Condition matching was repaired after the experiments: the old comparison key omitted framing and could overwrite a condition. The repaired code preserves fixed factors and rejects duplicate keys. In the 46 audited contrasts, 34 means changed by at most 0.04167; none changed whether its interval excluded zero. Marginal accuracies and single-condition receiver/head experiments are unaffected by this defect. The combined-model inference is a post-hoc summary.

