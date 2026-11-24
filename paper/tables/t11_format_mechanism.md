| Model\|format\|order | READ clean | READ blocked | READ best restored | APPLY clean | APPLY blocked | APPLY best restored | Binding top-8 vs random-8 (READ) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B\|codes\|book_first | 0.96 | 0.21 | 0.94 at L=18 | 0.62 | 0.00 | 0.19 | 0.56 vs 0.94: -0.38 [-0.53,-0.23] |
| Qwen3-4B\|codes\|rule_first | 0.79 | 0.48 | 0.75 at L=16 | 0.27 | 0.02 | 0.12 | 0.77 vs 0.79: -0.02 [-0.08,+0.04] |
| Qwen3-4B\|nl\|book_first | 0.81 | 0.29 | 0.83 at L=16 | 0.29 | 0.06 | 0.25 | 0.62 vs 0.81: -0.19 [-0.31,-0.06] |
| Qwen3-4B\|nl\|rule_first | 0.85 | 0.42 | 0.81 at L=16 | 0.46 | 0.17 | 0.44 | 0.77 vs 0.86: -0.09 [-0.17,-0.02] |
| Qwen3-8B\|codes\|book_first | 0.94 | 0.27 | 0.90 at L=16 | 0.46 | 0.04 | 0.25 | 0.58 vs 0.94: -0.35 [-0.48,-0.23] |
| Qwen3-8B\|codes\|rule_first | 0.83 | 0.42 | 0.71 at L=16 | 0.25 | 0.02 | 0.31 | 0.81 vs 0.83: -0.02 [-0.10,+0.04] |
| Qwen3-8B\|nl\|book_first | 0.75 | 0.65 | 0.83 at L=14 | 0.27 | 0.10 | 0.21 | 0.71 vs 0.73: -0.02 [-0.15,+0.10] |
| Qwen3-8B\|nl\|rule_first | 0.94 | 0.56 | 0.83 at L=16 | 0.33 | 0.12 | 0.35 | 0.83 vs 0.92: -0.09 [-0.19,-0.01] |

Later source blocked from the earlier one in all layers; restoration writes back clean later-source states after layer L.
