| Model | READ_DEC bf | READ_RULE bf | READ_RULE rf | APPLY bf | APPLY rf | APPLY stop-at-base | Leak bf | Leak rf |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 1.00 | 0.95 | 0.73 | 0.64 | 0.36 | 0.38 | 0/192 | 0/192 |
| Qwen3-1.7B | 0.99 | 0.87 | 0.48 | 0.19 | 0.04 | 0.84 | 7/192 | 5/192 |
| Qwen2.5-3B | 0.97 | 0.78 | 0.32 | 0.12 | 0.08 | 0.82 | 0/192 | 3/192 |
| Phi-3.5-mini | 0.76 | 0.80 | 0.38 | 0.66 | 0.02 | 0.63 | 29/192 | 0/192 |

Frozen models, 4-candidate log-prob probe; bf = codebook first, rf = rule first.
