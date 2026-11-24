# Comparison contract and primary references

All trainable methods use the same frozen Qwen checkpoint, native eager attention kernel, selected layer(s), supervised examples and target-only next-token CE. No method sees gold during generation. Default ranks approximately match VECTOR_PAIR's parameter count (nearest integer rank, capped at 128); actual counts and per-method ranks are saved. RCA is a low-capacity scalar reference, not capacity matched. Equal parameter count is not equal compute. The main preset has no hyperparameter search; a publication needs a preregistered, equal search budget on development data.

| Code name | Implemented operator | What the comparison asks |
|---|---|---|
| FULL | Frozen model, native eager attention | Does adaptation help at all? |
| NATIVE_SDPA | Frozen model, optimized HF SDPA | Systems reference; numerical differences can alter outputs |
| SCOPE_PROMPT | Frozen FULL plus explicit scope reminder | Can a prompt solve the problem? |
| FACTS_ONLY | Rule section removed | Information-deletion ablation, not a fair comparator on tasks requiring R |
| FACTS_MATCHED / ALT | Replace rule tokens using offsets, exact original input-token count and two filler streams | Does rule content affect NONAPPLY, with length/position controlled? |
| STATIC | Learned source-independent low-rank value residual | Capacity without conditioning |
| QUERY | Low-rank query-conditioned value residual | Exact equivalent post-aggregation operator |
| KEY | Source-conditioned low-rank value residual | Does query dependence add value? |
| SEPARABLE | Product of query and key feature gates; factorized aggregation | Strong efficient separable control |
| SCALAR_PAIR | Pairwise scalar gate on low-rank value branch | Is feature-specific conditioning needed? |
| VECTOR_PAIR | Pairwise feature gate on low-rank value branch, chunked over queries | Candidate under test |
| POST_GATE | Native SDPA output multiplied by 1+tanh(low-rank query projection) | Gated Attention-inspired operator control |
| RCA_ADAPT | Residual gamma * sum alpha * softplus(raw QK score) * V | RCA-inspired routing/gain control |
| DVA_ADAPT | Low-rank sum alpha * (query feature * key feature), projected to head output | DVA-inspired dynamic-value control |
| AUX_ATTENTION | A separate causal low-rank QK attention route and value branch | Extra attention capacity control |
| LORA_V / O / QK | Standard zero-initialized low-rank residual on the corresponding linear projections | Does ordinary parameter-efficient fine-tuning explain gains? |
| RESIDUAL | Bottleneck GELU residual after attention output projection | Non-attention bottleneck capacity control |

## Published work and fidelity

**Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free — NeurIPS 2025.** [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html). The paper studies a head-specific sigmoid after SDPA. POST_GATE here uses a low-rank, identity-normalized sigmoid equivalent (1+tanh(x)=2 sigmoid(2x)), and trains only this adapter on a frozen Qwen. This is an operator adaptation, not the original pretraining experiment.

**Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time — Findings ACL 2026.** [Official paper](https://aclanthology.org/2026.findings-acl.1824/). RCA_ADAPT retains scalar raw-QK rectification and unnormalized value gain. It learns a signed zero-initialized gamma, operates at the selected layers and uses no context-token annotation. The source method is training-free; this adaptation is not a reproduction of its complete protocol.

**Transformer Reconstructed with Dynamic Value Attention.** [Primary preprint](https://arxiv.org/abs/2512.22212). DVA_ADAPT implements a low-rank query/key Hadamard value residual. It does not reconstruct or pretrain the full DVA architecture.

**LoRA: Low-Rank Adaptation of Large Language Models — ICLR 2022.** [Primary paper](https://arxiv.org/abs/2106.09685). The linear residual BA is implemented directly; scaling is absorbed into the learned factors (alpha/r=1), B starts at zero, A random, no dropout. Target modules, parameter budget and local training protocol are documented here rather than borrowing the paper's reported numbers.

These comparisons produce new, local, same-data measurements. They do not compare our synthetic score to unrelated published leaderboard values. ASIDE, DRIP, official RCA inference settings, official Gated Attention training, external language benchmarks and natural-task transfer are NOT_RUN. They must not be listed as methods we defeated.
