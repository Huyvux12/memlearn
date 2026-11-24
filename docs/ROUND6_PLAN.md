# Round 6 — pre-registered predictions (written 23 Oct 2026, before the run)

**Goal.** Round 5 found that the binding reaches the reader heads through their keys. Round 6 sharpens and extends that finding:
- **G9.** The receiver result was obtained at the level of key/value groups. In grouped-query models, one group serves 2–4 query heads. Is the effect carried by the reader heads themselves, or by the other query heads that share their groups?
- **G10.** Does the key result replicate on fresh bundles?
- **G11.** Llama-3, the most common open family, was missing in round 5.

**Setup.**
- Code v2.3.0; one Colab job (`deploy/colab_round6/job.json`).
- **Llama-3 source.** Llama-3 is loaded from the ungated `unsloth` mirrors of `Llama-3.1-8B-Instruct` and `Llama-3.2-3B-Instruct`, which hold the same weights. The Llama 3.1/3.2 Community License applies. Llama-3.2's chat template writes the current date; it is pinned to Llama-3.1's default ("26 Jul 2024") so that prompts do not depend on the run date. All other models' prompts are unchanged.
- **Bundles.** The per-head runs use fresh bundles (`data_seed = 61`), and the heads are ranked on held-out bundles (`head_seed = 73`). As in round 5: codebook first, neutral framing, no distractor, APPLY_P and READ_RULE, 24 + 24 bundles.
- **Statistics.** Paired cluster bootstraps over bundles.

## G9 — keys per query head (`mech_path` part `heads`)

**Method.**
- In the blocked run (rule ↛ codebook, all layers), selected **query heads** receive the clean keys at the rule tokens. The other query heads in the same groups keep the blocked keys.
- Keys are expanded per query head inside the attention kernel. The KV cache is never modified. The tests check that giving clean keys to *every* query head of a group is identical to the group-level key restoration.

**Variants:**
- `kh_read{k}` for k = 1, 2, 4, 8, 16: the top-k reader heads;
- `kh_rest8`: the group-mates of the top-8 readers (their groups minus the readers);
- `kh_rand8`: 8 random query heads, 3 draws;
- the round-5 receiver variants (`kv_read8`, `k_read8`, `v_read8`, `kv_rand8`) for comparison.

**Models.**
- Codes: Qwen3-4B, Qwen3-8B, Phi-3.5, Mistral-7B, Gemma-2-9B, Llama-3.1-8B.
- Prose (nl): Qwen3-4B.
- Phi-3.5 has one query head per group, so `kh_read8` is the same as the group-level restoration there and `kh_rest8` is empty.

**Predictions** (READ_RULE):
- **P9a.** `kh_read8` beats `kh_rand8` by ≥ 0.20 in ≥ 4 of the 5 grouped-query codes models.
- **P9b.** `kh_read8` recovers ≥ 70 % of what the group-level `k_read8` recovers, measured from the blocked level, in ≥ 3 of 5 grouped-query codes models. If it does not, and `kh_rest8` recovers comparably, the group-mates carry the effect, and the paper reports the receiver at group level.
- **P9c (dose, descriptive).** Recovery grows with k from 1 to 16. We report the smallest k reaching half of the `k_read8` recovery.

## G10 — replication of the round-5 receiver result on fresh bundles

- **P10a.** Restoring keys only beats restoring values only (`k_read8 − v_read8` on READ_RULE, CI excludes 0) in ≥ 5 of the 6 codes models.
- **P10b.** `kv_read8 − kv_rand8` on READ_RULE is ≥ 0.15 with a CI excluding 0 in ≥ 5 of the 6 codes models.

## G11 — Llama-3

**Llama-3.1-8B.** Frozen probe, knockout, prose, circuit (`mech_e4`), and the full `mech_path` (sender, receiver, route, heads) on the round-5 bundles (seed 41), with L* = 14 (as for Mistral).

**Llama-3.2-3B.** Frozen probe, knockout, prose.

**Predictions:** as round 5 P8a–d.
- **P8a.** READ bf > APPLY bf.
- **P8b.** READ bf > READ rf.
- **P8c.** Knockout M1 < sham M2.
- **P8d.** On Llama-3.1-8B, the reader heads and the binding → reader path hold as in round 4.
- **Sender (P5b).** Attention > MLP holds for Llama-3.1-8B.
- **Weak models** (READ bf < 0.6) are reported but not counted for P8b–c.
