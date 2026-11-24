| Run | clean | blocked | all | attn | MLP | top-8 heads | top-32 | top-128 | random-32 | random-128 | top-32 + MLP |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codes_qwen3_4b | 0.69 | 0.00 | 0.23 | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 | 0.00 | 0.01 | 0.00 |
| codes_qwen3_8b | 0.67 | 0.08 | 0.29 | 0.04 | 0.00 | 0.06 | 0.08 | 0.06 | 0.08 | 0.06 | 0.04 |
| codes_phi35_mini | 0.79 | 0.04 | 0.56 | 0.33 | 0.04 | 0.17 | 0.25 | 0.12 | 0.07 | 0.15 | 0.25 |
| codes_mistral_7b | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| nl_qwen3_4b | 0.29 | 0.06 | 0.23 | 0.15 | 0.00 | 0.08 | 0.06 | 0.06 | 0.07 | 0.08 | 0.00 |
| nl_qwen3_8b | 0.27 | 0.10 | 0.12 | 0.10 | 0.02 | 0.10 | 0.10 | 0.08 | 0.09 | 0.08 | 0.02 |
| shelf_qwen3_4b | 0.04 | 0.02 | 0.04 | 0.00 | 0.00 | 0.02 | 0.02 | 0.02 | 0.01 | 0.00 | 0.00 |
| shelf_qwen3_8b | 0.25 | 0.04 | 0.15 | 0.08 | 0.06 | 0.10 | 0.10 | 0.08 | 0.06 | 0.07 | 0.06 |
| codes_llama31_8b | 0.52 | 0.02 | 0.15 | 0.02 | 0.02 | 0.02 | 0.00 | 0.00 | 0.02 | 0.10 | 0.02 |

APPLY_P accuracy. Rule tokens blocked from the codebook in every layer; after layer L* the clean-minus-blocked direct contributions of the listed components (layers 0..L*) are added back at the rule tokens. Random sets: mean of 3 draws. Heads ranked on held-out bundles.
