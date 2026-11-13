## E4-length - AUROC x100

| dataset | tag | LRR | entropy | likelihood | logrank | rank |
|---|---|---|---|---|---|---|
| SQuAD | 100w | 93.92 | 58.62 | 85.93 | 90.75 | 74.86 |
| SQuAD | 25w | 52.97 | 49.78 | 53.23 | 53.25 | 51.08 |
| SQuAD | 50w | 78.81 | 56.2 | 69.46 | 74.0 | 63.47 |
| SQuAD | E1 default (T=1, top-k=50, full length, fp32) | 97.83 | 55.36 | 91.55 | 95.12 | 82.01 |
| WritingPrompts | 100w | 94.85 | 42.51 | 93.36 | 95.0 | 84.1 |
| WritingPrompts | 25w | 53.73 | 50.2 | 54.04 | 53.95 | 51.92 |
| WritingPrompts | 50w | 83.09 | 49.92 | 83.08 | 84.77 | 71.13 |
| WritingPrompts | E1 default (T=1, top-k=50, full length, fp32) | 98.71 | 33.59 | 96.28 | 97.89 | 90.91 |
| XSum | 100w | 89.18 | 56.44 | 86.04 | 88.71 | 74.6 |
| XSum | 25w | 51.26 | 49.9 | 53.22 | 53.23 | 51.17 |
| XSum | 50w | 70.98 | 54.94 | 72.42 | 73.89 | 61.86 |
| XSum | E1 default (T=1, top-k=50, full length, fp32) | 92.74 | 53.87 | 89.55 | 92.08 | 80.29 |

## E4-length - TPR@1%FPR x100

| dataset | tag | LRR | entropy | likelihood | logrank | rank |
|---|---|---|---|---|---|---|
| SQuAD | 100w | 34.0 | 3.0 | 35.0 | 42.0 | 24.0 |
| SQuAD | 25w | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 |
| SQuAD | 50w | 8.0 | 4.0 | 3.0 | 6.0 | 6.0 |
| SQuAD | E1 default (T=1, top-k=50, full length, fp32) | 58.0 | 0.0 | 15.0 | 25.0 | 45.0 |
| WritingPrompts | 100w | 35.0 | 0.0 | 31.0 | 47.0 | 44.0 |
| WritingPrompts | 25w | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| WritingPrompts | 50w | 12.0 | 1.0 | 2.0 | 4.0 | 16.0 |
| WritingPrompts | E1 default (T=1, top-k=50, full length, fp32) | 81.0 | 0.0 | 49.0 | 63.0 | 60.0 |
| XSum | 100w | 40.0 | 4.0 | 23.0 | 36.0 | 16.0 |
| XSum | 25w | 1.0 | 2.0 | 2.0 | 1.0 | 1.0 |
| XSum | 50w | 9.0 | 3.0 | 8.0 | 9.0 | 7.0 |
| XSum | E1 default (T=1, top-k=50, full length, fp32) | 57.0 | 3.0 | 37.0 | 51.0 | 17.0 |
