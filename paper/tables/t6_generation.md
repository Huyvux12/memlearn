| Model | Generated items | Invalid rate | Agreement with log-prob | Gen READ_RULE bf | Gen APPLY bf | Gen READ_RULE rf | Gen leak (NONAPPLY) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 423 | 0.26 | 1.00 | 0.98 | 0.61 | 0.81 | 0/6 |
| Qwen3-1.7B | 423 | 0.60 | 0.98 | 0.80 | nan | 0.54 | 0/0 |
| Qwen2.5-3B | 423 | 0.67 | 0.90 | 0.86 | nan | 0.17 | 0/0 |
| Phi-3.5-mini | 423 | 0.61 | 0.98 | 0.74 | nan | 0.36 | 0/0 |

Greedy free generation (no teacher forcing) on every 4th item; accuracies over valid parses.
