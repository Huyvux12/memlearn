| Model | READ_DEC bf | READ_RULE bf | READ_RULE rf | APPLY bf | APPLY rf | APPLY stop-at-base | Leak bf | Leak rf |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-9B | 1.00 | 0.92 | 0.71 | 0.29 | 0.15 | 0.76 | 0/192 | 0/192 |
| Gemma-2-2B | 0.97 | 0.71 | 0.10 | 0.06 | 0.01 | 0.96 | 5/192 | 0/192 |
| Mistral-7B | 1.00 | 0.65 | 0.45 | 0.10 | 0.13 | 0.84 | 0/192 | 4/192 |
| Llama-3.1-8B | 0.94 | 0.73 | 0.35 | 0.57 | 0.22 | 0.52 | 17/192 | 3/192 |
| Llama-3.2-3B | 1.00 | 0.17 | 0.01 | 0.15 | 0.02 | 0.89 | 8/192 | 2/192 |

Round 5-6 families (Gemma-2, Mistral, Llama-3), frozen probe (codes).
