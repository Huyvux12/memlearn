| Run | clean | blocked | all | attn | MLP | top-8 heads | top-32 | top-128 | random-32 | random-128 | top-32 + MLP |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codes_qwen3_4b | 0.96 | 0.19 | 0.90 | 0.62 | 0.31 | 0.54 | 0.58 | 0.08 | 0.26 | 0.42 | 0.54 |
| codes_qwen3_8b | 1.00 | 0.21 | 0.88 | 0.46 | 0.29 | 0.38 | 0.38 | 0.23 | 0.23 | 0.30 | 0.42 |
| codes_phi35_mini | 0.92 | 0.52 | 0.77 | 0.65 | 0.46 | 0.56 | 0.58 | 0.54 | 0.54 | 0.54 | 0.50 |
| codes_mistral_7b | 0.60 | 0.17 | 0.50 | 0.38 | 0.25 | 0.25 | 0.27 | 0.29 | 0.23 | 0.26 | 0.38 |
| nl_qwen3_4b | 0.81 | 0.29 | 0.83 | 0.40 | 0.10 | 0.38 | 0.42 | 0.29 | 0.31 | 0.32 | 0.33 |
| nl_qwen3_8b | 0.75 | 0.65 | 0.83 | 0.77 | 0.65 | 0.75 | 0.79 | 0.67 | 0.66 | 0.67 | 0.77 |
| shelf_qwen3_4b | 0.88 | 0.42 | 0.85 | 0.56 | 0.27 | 0.52 | 0.48 | 0.50 | 0.44 | 0.44 | 0.50 |
| shelf_qwen3_8b | 0.71 | 0.69 | 0.75 | 0.69 | 0.73 | 0.73 | 0.67 | 0.75 | 0.69 | 0.67 | 0.73 |
| codes_llama31_8b | 0.85 | 0.02 | 0.15 | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 | 0.03 | 0.06 | 0.00 |

READ_RULE accuracy. Rule tokens blocked from the codebook in every layer; after layer L* the clean-minus-blocked direct contributions of the listed components (layers 0..L*) are added back at the rule tokens. Random sets: mean of 3 draws. Heads ranked on held-out bundles.
