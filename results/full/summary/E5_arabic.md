## E5 - AUROC x100

| exp | dataset | generator | scorer | LRR | entropy | likelihood | logrank | rank |
|---|---|---|---|---|---|---|---|---|
| E5-arabic | Arabic Wikipedia | AraGPT2-M | AraGPT2-M | 99.45 | 21.77 | 98.54 | 99.11 | 89.3 |
| E5-arabic | Arabic Wikipedia | Qwen2.5-1.5B | Qwen2.5-1.5B | 86.26 | 63.28 | 65.42 | 71.58 | 86.37 |
| E5-control | XSum | Qwen2.5-1.5B | Qwen2.5-1.5B | 91.93 | 56.78 | 80.52 | 86.29 | 75.32 |
| E5-cross | Arabic Wikipedia | AraGPT2-M | Qwen2.5-1.5B | 63.82 | 52.38 | 58.99 | 60.74 | 56.93 |
| E5-cross | Arabic Wikipedia | Qwen2.5-1.5B | AraGPT2-M | 23.87 | 78.96 | 17.12 | 15.67 | 13.73 |

## E5 - TPR@1%FPR x100

| exp | dataset | gen | model | LRR | entropy | likelihood | logrank | rank |
|---|---|---|---|---|---|---|---|---|
| E5-arabic | arwiki | AraGPT2-M | AraGPT2-M | 91.0 | 0.0 | 83.0 | 92.0 | 61.0 |
| E5-arabic | arwiki | Qwen2.5-1.5B | Qwen2.5-1.5B | 11.0 | 14.0 | 10.0 | 12.0 | 14.0 |
| E5-control | xsum | Qwen2.5-1.5B | Qwen2.5-1.5B | 35.0 | 10.0 | 19.0 | 23.0 | 23.0 |
| E5-cross | arwiki | AraGPT2-M | Qwen2.5-1.5B | 6.0 | 3.0 | 7.0 | 8.0 | 6.0 |
| E5-cross | arwiki | Qwen2.5-1.5B | AraGPT2-M | 0.0 | 12.0 | 0.0 | 0.0 | 0.0 |

## LRR with 95% CI

| exp | dataset | gen | model | auroc | auroc_lo | auroc_hi |
|---|---|---|---|---|---|---|
| E5-control | xsum | Qwen2.5-1.5B | Qwen2.5-1.5B | 91.93 | 88.17 | 95.22 |
| E5-arabic | arwiki | Qwen2.5-1.5B | Qwen2.5-1.5B | 86.26 | 80.96 | 91.12 |
| E5-cross | arwiki | Qwen2.5-1.5B | AraGPT2-M | 23.87 | 18.41 | 29.05 |
| E5-cross | arwiki | AraGPT2-M | Qwen2.5-1.5B | 63.82 | 56.78 | 70.54 |
| E5-arabic | arwiki | AraGPT2-M | AraGPT2-M | 99.45 | 98.83 | 99.85 |
