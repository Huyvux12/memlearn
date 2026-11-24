# Round 5 — pre-registered predictions (written 23 Oct 2026, before the run)

**Goal:** close the gaps reviewers are most likely to raise against the round-4 circuit:
- G5: 32 binding heads are not sufficient, so it is unclear what writes the binding into the rule tokens (heads vs MLPs, few vs many heads);
- G6: the reader heads are necessary for APPLY, but it is not shown that the binding reaches them through their keys/values at the rule tokens;
- G7: Qwen3-8B reads prose rules with little dependence on the codebook-first block (round 4: READ 0.75 → 0.65), and blocking only the top-32 binding heads hurt *more* than blocking every head (0.56);
- G8: families beyond Qwen, Phi, OLMo and Granite (Llama is unavailable: access not granted).

**Setup.**
- Code v2.2.0; one Colab job (`deploy/colab_round5/job.json`).
- New module `memlearn_attention/mech_path.py`. It uses **fresh bundles** (`data_seed = 41`, both modes APPLY_P and READ_RULE, 24 + 24 bundles). Binding and reader heads are ranked on **held-out bundles** (`head_seed = 53`), never on the evaluated ones; this is new relative to round 4.
- Codebook first, neutral framing, no distractor. Scoring uses the 4-candidate log-prob probe; CIs are paired cluster bootstraps over bundles.
- "Blocked" means the rule tokens (later source) are blocked from the whole codebook in every layer and head, as in E4.

## G5 — what writes the binding (sender side; decomposition at layer L*)

**Method.** Run the blocked prompt. After layer L*, add to the residual stream at the rule tokens the clean-minus-blocked *direct contribution* of a chosen set of components of layers 0..L*:
- a head's contribution is its slice of the `o_proj` input times its slice of `o_proj`;
- an MLP's contribution is its output.

With every component (`dec_all`), this rebuilds the clean residual exactly. That is checked in the tests and only valid for pre-norm decoders, so it runs on Qwen3, Phi-3.5 and Mistral, not on Gemma-2.

- **L\*:** 16 for Qwen3 (the shared restoration window, round 4 G1), 12 for Phi-3.5 (round 3), 14 for Mistral (the same relative depth as Qwen3).
- **Head sets:** the top-k binding heads among layers ≤ L* (k = 8, 32, 128), and random-k from the same layers (k = 32 and 128, 3 draws each).

Recovery is measured as the fraction of the clean − blocked gap.

**Predictions** (codes runs: Qwen3-4B, Qwen3-8B, Phi-3.5, Mistral-7B):
- **P5a (sanity, must hold or G5 is void).** `dec_all` brings READ_RULE within 0.10 of clean.
- **P5b.** Attention contributions (`dec_attn`) recover more of the READ_RULE gap than MLP contributions (`dec_mlp`), in ≥ 3/4 models.
- **P5c.** `dec_top32` recovers READ_RULE by ≥ 0.20 more than `dec_rand32` in ≥ 3/4 models. Recovery grows from k = 8 to k = 128.
- **P5d.** If `dec_top32_mlp − dec_top32` ≥ 0.15 on READ_RULE, the round-4 insufficiency of 32 heads is (partly) explained by MLP processing at the rule tokens. If not, it is explained by heads outside the top 32 (judged by `dec_top128`). Both outcomes are reported.

## G6 — how the reader heads use the binding (receiver side)

**Method.** In the blocked run, the key/value groups of the top-8 reader heads (grouped-query models restore whole groups) get their **clean keys and values** at the rule tokens. This is done by replacing the `k_proj`/`v_proj` outputs, or the K/V slices of a fused `qkv_proj`; the KV cache then holds them for the answer tokens. Variants restore keys only, values only, or both.

**Control:** 8 random heads from all layers, mapped to their groups, 3 draws.

**Predictions** (codes runs, 5 models including Gemma-2-9B):
- **P6a.** `kv_read8` raises APPLY_P by ≥ 0.20 more than `kv_rand8` in ≥ 3/5 models.
- **P6b (two-sided, no direction predicted).** Keys only vs values only: which one carries the recovery.

## G7 — the prose route in Qwen3-8B

**Hypothesis H-direct.** Answering READ_RULE does not need the codebook: the queried value is already written in the JSON prefix (`"decoded_value":"Amber"`). In prose, a rule entry ("room Amber moves to room Cedar") can be found by matching that word directly.
- Qwen3-8B uses this direct route in prose, so removing the codebook completely costs little.
- A *partial* block (only the top binding heads) leaves corrupted bindings in the rule tokens that mislead the reader heads.

**Runs:** `nl` and `nl_shelf` on Qwen3-4B and Qwen3-8B, plus all codes runs for comparison.

**Variants:**
- `blocked`: the whole codebook;
- `blocked_vals`: codebook value tokens only, all heads;
- `bind_top32`: value tokens, top-32 binding heads only;
- `bind_rand32`: the same with 32 random heads.

**Measurements:** accuracy, and the top-8 reader heads' attention to the queried entry under each variant.

**Predictions:**
- **P7a.** For Qwen3-8B prose (both templates), READ_RULE under `bind_top32` is lower than under `blocked` (paired difference < 0, CI excludes 0). For Qwen3-4B (both formats) and Qwen3-8B codes, it is not lower.
- **P7b.** For Qwen3-8B prose, the reader heads' attention to the queried entry under `blocked` stays ≥ 70 % of clean. In codes (Qwen3-4B, Qwen3-8B) it falls below 50 % of clean.
- **P7c.** For Qwen3-8B prose, `bind_top32` lowers that attention more than `blocked` does.

*If P7a–c fail:* the round-4 pattern was specific to those bundles, and the paper says so.

## G8 — more families (codes and prose; frozen probe and knockout as in round 4 G3; circuit replication)

**Models:** Gemma-2-9B-it, Gemma-2-2B-it (bfloat16; the chat template has no system role, so the contract opens the user turn), Mistral-7B-Instruct-v0.3.

**Predictions** (as round 4 G3, per model):
- **P8a.** READ bf > APPLY bf (CI excludes 0).
- **P8b.** READ bf > READ rf.
- **P8c.** Knockout M1 lowers READ_RULE more than its sham M2.
- **P8d (circuit).** On Gemma-2-9B and Mistral-7B, P2b (reader heads) and P2c (binding → reader path) hold as in round 4.

Weak models (READ bf < 0.6) are reported but not counted for P8b–c.

## Not planned

Llama-3 (gated; the account has no access yet); free generation; adapters.
