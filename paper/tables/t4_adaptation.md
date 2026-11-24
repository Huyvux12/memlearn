| Setting | Method | Seeds | APPLY bf | APPLY rf | APPLY rf (seed range) | READ_RULE | Leak (NONAPPLY) | Leak rate |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B, 4 layers, lr 5e-4, train bf | Separable | 5 | 0.92 | 0.48 | 0.38-0.56 | 0.90 | 8/480 | 0.02 |
| Qwen3-4B, 4 layers, lr 5e-4, train bf | Vector pair | 5 | 0.90 | 0.43 | 0.42-0.46 | 0.89 | 6/480 | 0.01 |
| Qwen3-4B, 4 layers, lr 5e-4, train bf | LoRA-qkvo | 5 | 0.96 | 0.63 | 0.33-0.75 | 0.89 | 9/480 | 0.02 |
| Qwen3-4B, 4 layers, lr 1e-3, train bf | Separable | 5 | 0.93 | 0.52 | 0.21-0.62 | 0.89 | 10/480 | 0.02 |
| Qwen3-4B, 4 layers, lr 1e-3, train bf | Vector pair | 5 | 0.88 | 0.55 | 0.40-0.67 | 0.89 | 11/480 | 0.02 |
| Qwen3-4B, 4 layers, lr 1e-3, train bf | LoRA-qkvo | 5 | 0.98 | 0.61 | 0.50-0.79 | 0.94 | 5/480 | 0.01 |
| Qwen3-4B, 4 layers, lr 2e-3, train bf | Separable | 5 | 0.95 | 0.68 | 0.33-0.98 | 0.93 | 23/480 | 0.05 |
| Qwen3-4B, 4 layers, lr 2e-3, train bf | Vector pair | 5 | 1.00 | 0.52 | 0.33-0.67 | 0.91 | 14/480 | 0.03 |
| Qwen3-4B, 4 layers, lr 2e-3, train bf | LoRA-qkvo | 5 | 0.83 | 0.29 | 0.17-0.54 | 0.82 | 58/480 | 0.12 |
| Qwen3-4B, 4 layers, lr 5e-4, train bf+rf | Separable | 5 | 0.90 | 0.80 | 0.69-0.94 | 0.96 | 7/480 | 0.01 |
| Qwen3-4B, 4 layers, lr 5e-4, train bf+rf | Vector pair | 5 | 0.83 | 0.70 | 0.62-0.81 | 0.93 | 4/480 | 0.01 |
| Qwen3-4B, 4 layers, lr 5e-4, train bf+rf | LoRA-qkvo | 5 | 0.95 | 0.94 | 0.92-0.98 | 0.98 | 54/480 | 0.11 |
| Qwen3-4B, 1 layer, lr 2e-3, train bf | Query | 5 | 0.90 | 0.52 | 0.27-0.60 | 0.83 | 29/480 | 0.06 |
| Qwen3-4B, 1 layer, lr 2e-3, train bf | Separable | 5 | 0.89 | 0.52 | 0.29-0.65 | 0.81 | 27/480 | 0.06 |
| Qwen3-4B, 1 layer, lr 2e-3, train bf | Vector pair | 5 | 0.89 | 0.54 | 0.40-0.71 | 0.80 | 43/480 | 0.09 |
| Qwen3-4B, 1 layer, lr 2e-3, train bf | LoRA-v | 5 | 0.76 | 0.27 | 0.08-0.40 | 0.81 | 45/480 | 0.09 |
| Phi-3.5-mini, 4 layers, lr 5e-4, train bf | Query | 5 | 0.88 | 0.14 | 0.08-0.19 | 0.80 | 28/480 | 0.06 |
| Phi-3.5-mini, 4 layers, lr 5e-4, train bf | Separable | 5 | 0.93 | 0.10 | 0.04-0.17 | 0.81 | 25/480 | 0.05 |
| Phi-3.5-mini, 4 layers, lr 5e-4, train bf | Vector pair | 5 | 0.85 | 0.14 | 0.08-0.19 | 0.79 | 26/480 | 0.05 |
| Phi-3.5-mini, 4 layers, lr 5e-4, train bf | LoRA-qkvo | 5 | 0.97 | 0.07 | 0.04-0.19 | 0.81 | 19/480 | 0.04 |

Adapters on frozen backbones, parameter-matched, 256 steps; evaluation neutral framing x {none,q} x {bf,rf}. Training never sees rf unless stated.
