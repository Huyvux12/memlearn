| Run | READ clean / blocked | group keys (k_read8) | reader heads only, k = 1/2/4/8/16 | group-mates of top-8 | random 8 heads | P9a top-8 - random-8 | top-8 - group-mates | P9b top-8 / group recovery | P9c smallest k reaching half |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codes_qwen3_4b | 0.96 / 0.27 | 0.67 | 0.42 / 0.50 / 0.73 / 0.73 / 0.79 | 0.27 | 0.32 | +0.41 [+0.28,+0.54] | +0.46 [+0.31,+0.60] | 1.16 | 2 |
| codes_qwen3_8b | 1.00 / 0.15 | 0.77 | 0.46 / 0.67 / 0.69 / 0.73 / 0.73 | 0.19 | 0.27 | +0.46 [+0.34,+0.58] | +0.54 [+0.40,+0.69] | 0.93 | 1 |
| codes_phi35_mini | 0.92 / 0.52 | 0.83 | 0.54 / 0.67 / 0.77 / 0.83 / 0.90 | 0.52 | 0.56 | +0.27 [+0.15,+0.40] | +0.31 [+0.19,+0.44] | 1.00 | 4 |
| codes_mistral_7b | 0.85 / 0.31 | 0.54 | 0.33 / 0.42 / 0.44 / 0.48 / 0.44 | 0.33 | 0.31 | +0.17 [+0.04,+0.30] | +0.15 [+0.00,+0.29] | 0.73 | 4 |
| codes_gemma2_9b | 0.90 / 0.56 | 0.90 | 0.58 / 0.67 / 0.77 / 0.90 / 0.92 | 0.50 | 0.51 | +0.38 [+0.24,+0.53] | +0.40 [+0.25,+0.54] | 1.00 | 4 |
| codes_llama31_8b | 0.71 / 0.02 | 0.27 | 0.02 / 0.06 / 0.15 / 0.17 / 0.38 | 0.02 | 0.01 | +0.15 [+0.06,+0.26] | +0.15 [+0.06,+0.25] | 0.58 | 4 |
| nl_qwen3_4b | 0.81 / 0.46 | 0.75 | 0.67 / 0.71 / 0.79 / 0.83 / 0.81 | 0.40 | 0.51 | +0.32 [+0.20,+0.44] | +0.44 [+0.31,+0.58] | 1.29 | 1 |

READ_RULE in the blocked run with clean keys at the rule tokens for single query heads only (their group-mates keep blocked keys). Fresh bundles (seed 61); heads ranked on held-out bundles (seed 73).
