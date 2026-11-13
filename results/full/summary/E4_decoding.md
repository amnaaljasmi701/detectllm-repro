## E4-decoding - AUROC x100

| dataset | tag | LRR | entropy | likelihood | logrank | rank |
|---|---|---|---|---|---|---|
| XSum | E1 default (T=1, top-k=50, full length, fp32) | 92.74 | 53.87 | 89.55 | 92.08 | 80.29 |
| XSum | T0.7 | 99.78 | 52.94 | 99.61 | 99.79 | 81.43 |
| XSum | pure | 23.08 | 75.67 | 29.84 | 27.38 | 27.8 |
| XSum | topp0.96 | 38.27 | 63.18 | 60.13 | 56.01 | 63.68 |

## E4-decoding - TPR@1%FPR x100

| dataset | tag | LRR | entropy | likelihood | logrank | rank |
|---|---|---|---|---|---|---|
| XSum | E1 default (T=1, top-k=50, full length, fp32) | 57.0 | 3.0 | 37.0 | 51.0 | 17.0 |
| XSum | T0.7 | 99.0 | 3.0 | 98.0 | 99.0 | 26.0 |
| XSum | pure | 0.0 | 14.0 | 1.0 | 1.0 | 0.0 |
| XSum | topp0.96 | 1.0 | 11.0 | 9.0 | 6.0 | 0.0 |
