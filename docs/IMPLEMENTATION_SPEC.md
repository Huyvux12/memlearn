# MEMLEARN Attention Lab 1.0 — implementation contract

Authority: MEMLEARN_ATTENTION_RESEARCH_GAP_V2.md and validation protocol V1, approved in conversation. This package is a new internal-attention experiment, unrelated to R21 agent code.

## Binding choices
- Frozen Qwen3 backbone, transformers 4.57.1, eager attention; no agent, external memory controller, gold routing or action correction.
- Gate at selected self-attention layers receives only native Q/K/V activations. Normal cross-entropy trains adapter parameters. Source/query text is ordinary model input.
- Per-head low-rank value branch: delta=sum_i alpha_ti U[g(q_t,k_i) * D v_i]. U starts at zero: exact identity before training.
- Operators: vector_pair, scalar_pair, separable, query, key, static; post-SDPA gating, RCA-inspired scalar reweighting, DVA-inspired dynamic value, LoRA V/O/QK and residual bottleneck capacity controls. Paper-inspired operators are adaptations, never claimed original paper reproductions.
- Q/K gates see post-RoPE normalized query/key; document this limitation. GQA repeats KV exactly; native eager mask retained. Pairwise implementation chunks query positions to bound temporary B*H*T*S*r memory. No attention-weight suppression by task labels.
- Neutral contract identical across trainable methods. Scope prompt and facts-only are explicit prompt controls, not purported new methods.

## Data and independent oracle
- Random nonce codebook K and non-identity permutation rule R. READ returns base and hypothetical R(base); APPLY returns R(K(x)); NONAPPLY returns K(x). MIXED requests all three with a permuted order.
- Independent fresh completion for each query, shared source prefix. No dialogue carryover or claim of temporal reinstatement in this release.
- Four training cue families, two development-OOD and two test-OOD families, with disjoint codebooks and seed-derived IDs. Development/test source data not used for optimizer or hyperparameter selection.
- JSON strict object equality plus decomposed field accuracy. Wrong-rule fingerprint distinct from unrelated failures. Parse error and output limit stay errors, never silently repaired.

## Scientific outputs
- Per-mode accuracy, per-bundle JSS, MIXED joint, rule leakage, READ base/rule recall, paired negative-transfer/rescue, train loss/params/compute, tokens, prefill/decode/latency/VRAM. Paired cluster bootstrap over bundles; seed-stratified results and no claim of power from a tiny pilot.
- Per-RQ technical validity, coverage, evidence and constrained conclusion. Missing baselines are NOT_RUN; unsuccessful generations are valid measurements. No success admission gate. Incomplete runs keep completed paired panels usable.
- RQ1 algebra and identity checks; RQ2 selective behavior vs FULL and prompt controls; RQ3 conditioned gate vs trained query/separable/scalar/capacity controls; RQ4 OOD, cost and multi-seed evidence.
- Native failure, trained repair and mechanism are separate. This version logs tensor algebra checks but does not claim causal localization from adapter success. No hidden-state probe interpreted as knowledge proof.

## Presets
- smoke: tiny random Qwen, software integration only; never scientific model results.
- pilot30: one seed, 24 train bundles, 4 dev bundles, 6 ID + 6 OOD test bundles, four queries each; FULL, SCOPE_PROMPT, FACTS_ONLY, QUERY and VECTOR_PAIR. Budget truncation explicit, panel evaluation round-robin. It screens trainability and contrast; cannot settle RQ3/RQ4 fully.
- research: full comparison set, 3 seeds, 512 train /128 dev-ID /64 dev-OOD /256 test-ID /256 test-OOD. GPU time not bounded to 30 minutes. Locked data/model/config/source hashes and resumable individual train steps/requests.
- A hard process watchdog terminates after wall budget and exports partial results. Budget includes setup once notebook timer starts. Download speed is not guaranteed.

## Verification
Actual CPU PyTorch tests include algebra, causal mask, GQA, cache decode equality, no future leakage, frozen-backbone gradients and adapter round trip with a tiny randomly initialized Qwen3. They certify software only. Pretrained GPU efficacy must be measured by user on Kaggle.
