## E1 - AUROC x100, ours

| dataset | method | GPT2-xl | Neo-2.7B | OPT-2.7B |
|---|---|---|---|---|
| SQuAD | LRR | 97.83 | 97.23 | 95.62 |
| SQuAD | entropy | 55.36 | 55.09 | 52.75 |
| SQuAD | likelihood | 91.55 | 85.7 | 87.4 |
| SQuAD | logrank | 95.12 | 90.68 | 91.47 |
| SQuAD | rank | 82.01 | 79.74 | 80.75 |
| WritingPrompts | LRR | 98.71 | 97.38 | 97.32 |
| WritingPrompts | entropy | 33.59 | 28.28 | 42.56 |
| WritingPrompts | likelihood | 96.28 | 95.15 | 92.92 |
| WritingPrompts | logrank | 97.89 | 96.29 | 94.76 |
| WritingPrompts | rank | 90.91 | 85.53 | 87.45 |
| XSum | LRR | 92.74 | 95.27 | 91.88 |
| XSum | entropy | 53.87 | 59.58 | 45.96 |
| XSum | likelihood | 89.55 | 86.76 | 88.79 |
| XSum | logrank | 92.08 | 91.24 | 91.61 |
| XSum | rank | 80.29 | 77.85 | 75.17 |

## E1 - AUROC x100, paper (Table 1)

| dataset | method | GPT2-xl | Neo-2.7B | OPT-2.7B |
|---|---|---|---|---|
| SQuAD | LRR | 97.42 | 95.74 | 95.89 |
| SQuAD | entropy | 57.97 | 58.48 | 53.29 |
| SQuAD | likelihood | 90.72 | 84.18 | 87.84 |
| SQuAD | logrank | 94.33 | 89.52 | 91.76 |
| SQuAD | rank | 83.46 | 79.77 | 81.85 |
| WritingPrompts | LRR | 98.34 | 98.02 | 96.45 |
| WritingPrompts | entropy | 36.45 | 34.07 | 39.75 |
| WritingPrompts | likelihood | 96.71 | 95.63 | 95.05 |
| WritingPrompts | logrank | 98.02 | 97.15 | 96.32 |
| WritingPrompts | rank | 87.62 | 82.79 | 83.89 |
| XSum | LRR | 93.47 | 92.24 | 88.7 |
| XSum | entropy | 56.78 | 55.14 | 50.34 |
| XSum | likelihood | 89.16 | 87.69 | 86.98 |
| XSum | logrank | 91.75 | 90.79 | 89.18 |
| XSum | rank | 79.79 | 77.87 | 76.07 |

## E1 - full rows

| dataset | model | method | ours | 95% CI | paper | diff | TPR@1%FPR | TPR@5%FPR | n |
|---|---|---|---|---|---|---|---|---|---|
| SQuAD | GPT2-xl | likelihood | 91.55 | [87.8, 94.8] | 90.72 | 0.83 | 15.0 | 48.0 | 100 |
| SQuAD | GPT2-xl | rank | 82.01 | [78.37, 86.2] | 83.46 | -1.45 | 45.0 | 55.0 | 100 |
| SQuAD | GPT2-xl | logrank | 95.12 | [92.11, 97.44] | 94.33 | 0.79 | 25.0 | 73.0 | 100 |
| SQuAD | GPT2-xl | entropy | 55.36 | [50.0, 60.88] | 57.97 | -2.61 | 0.0 | 7.0 | 100 |
| SQuAD | GPT2-xl | LRR | 97.83 | [95.92, 99.32] | 97.42 | 0.41 | 58.0 | 91.0 | 100 |
| WritingPrompts | GPT2-xl | likelihood | 96.28 | [93.84, 98.38] | 96.71 | -0.43 | 49.0 | 66.0 | 100 |
| WritingPrompts | GPT2-xl | rank | 90.91 | [87.7, 93.94] | 87.62 | 3.29 | 60.0 | 67.0 | 100 |
| WritingPrompts | GPT2-xl | logrank | 97.89 | [96.28, 99.23] | 98.02 | -0.13 | 63.0 | 82.0 | 100 |
| WritingPrompts | GPT2-xl | entropy | 33.59 | [27.26, 40.6] | 36.45 | -2.86 | 0.0 | 0.0 | 100 |
| WritingPrompts | GPT2-xl | LRR | 98.71 | [97.6, 99.62] | 98.34 | 0.37 | 81.0 | 94.0 | 100 |
| XSum | GPT2-xl | likelihood | 89.55 | [86.01, 93.28] | 89.16 | 0.39 | 37.0 | 52.0 | 100 |
| XSum | GPT2-xl | rank | 80.29 | [76.33, 84.99] | 79.79 | 0.5 | 17.0 | 51.0 | 100 |
| XSum | GPT2-xl | logrank | 92.08 | [88.75, 95.34] | 91.75 | 0.33 | 51.0 | 61.0 | 100 |
| XSum | GPT2-xl | entropy | 53.87 | [48.92, 59.47] | 56.78 | -2.91 | 3.0 | 12.0 | 100 |
| XSum | GPT2-xl | LRR | 92.74 | [89.27, 96.14] | 93.47 | -0.73 | 57.0 | 73.0 | 100 |
| SQuAD | Neo-2.7B | likelihood | 85.7 | [80.71, 89.82] | 84.18 | 1.52 | 18.0 | 25.0 | 100 |
| SQuAD | Neo-2.7B | rank | 79.74 | [75.75, 83.91] | 79.77 | -0.03 | 38.0 | 50.0 | 100 |
| SQuAD | Neo-2.7B | logrank | 90.68 | [86.7, 94.0] | 89.52 | 1.16 | 29.0 | 46.0 | 100 |
| SQuAD | Neo-2.7B | entropy | 55.09 | [48.67, 61.1] | 58.48 | -3.39 | 2.0 | 7.0 | 100 |
| SQuAD | Neo-2.7B | LRR | 97.23 | [95.42, 98.75] | 95.74 | 1.49 | 69.0 | 85.0 | 100 |
| WritingPrompts | Neo-2.7B | likelihood | 95.15 | [92.02, 97.8] | 95.63 | -0.48 | 24.0 | 66.0 | 100 |
| WritingPrompts | Neo-2.7B | rank | 85.53 | [81.73, 89.61] | 82.79 | 2.74 | 52.0 | 64.0 | 100 |
| WritingPrompts | Neo-2.7B | logrank | 96.29 | [93.58, 98.46] | 97.15 | -0.86 | 36.0 | 69.0 | 100 |
| WritingPrompts | Neo-2.7B | entropy | 28.28 | [21.5, 35.26] | 34.07 | -5.79 | 0.0 | 0.0 | 100 |
| WritingPrompts | Neo-2.7B | LRR | 97.38 | [94.98, 99.09] | 98.02 | -0.64 | 64.0 | 87.0 | 100 |
| XSum | Neo-2.7B | likelihood | 86.76 | [82.73, 90.74] | 87.69 | -0.93 | 22.0 | 36.0 | 100 |
| XSum | Neo-2.7B | rank | 77.85 | [73.99, 82.26] | 77.87 | -0.02 | 20.0 | 47.0 | 100 |
| XSum | Neo-2.7B | logrank | 91.24 | [88.06, 94.29] | 90.79 | 0.45 | 41.0 | 52.0 | 100 |
| XSum | Neo-2.7B | entropy | 59.58 | [54.04, 65.26] | 55.14 | 4.44 | 5.0 | 12.0 | 100 |
| XSum | Neo-2.7B | LRR | 95.27 | [92.86, 97.53] | 92.24 | 3.03 | 56.0 | 74.0 | 100 |
| SQuAD | OPT-2.7B | likelihood | 87.4 | [83.14, 91.3] | 87.84 | -0.44 | 24.0 | 41.0 | 100 |
| SQuAD | OPT-2.7B | rank | 80.75 | [76.68, 85.11] | 81.85 | -1.1 | 32.0 | 46.0 | 100 |
| SQuAD | OPT-2.7B | logrank | 91.47 | [87.97, 94.47] | 91.76 | -0.29 | 32.0 | 52.0 | 100 |
| SQuAD | OPT-2.7B | entropy | 52.75 | [46.88, 58.68] | 53.29 | -0.54 | 3.0 | 5.0 | 100 |
| SQuAD | OPT-2.7B | LRR | 95.62 | [93.23, 97.58] | 95.89 | -0.27 | 54.0 | 66.0 | 100 |
| WritingPrompts | OPT-2.7B | likelihood | 92.92 | [89.19, 96.28] | 95.05 | -2.13 | 17.0 | 52.0 | 100 |
| WritingPrompts | OPT-2.7B | rank | 87.45 | [83.44, 91.74] | 83.89 | 3.56 | 46.0 | 69.0 | 100 |
| WritingPrompts | OPT-2.7B | logrank | 94.76 | [91.62, 97.41] | 96.32 | -1.56 | 28.0 | 63.0 | 100 |
| WritingPrompts | OPT-2.7B | entropy | 42.56 | [35.55, 49.47] | 39.75 | 2.81 | 0.0 | 0.0 | 100 |
| WritingPrompts | OPT-2.7B | LRR | 97.32 | [95.16, 98.93] | 96.45 | 0.87 | 51.0 | 88.0 | 100 |
| XSum | OPT-2.7B | likelihood | 88.79 | [84.76, 92.71] | 86.98 | 1.81 | 17.0 | 50.0 | 100 |
| XSum | OPT-2.7B | rank | 75.17 | [71.63, 79.74] | 76.07 | -0.9 | 13.0 | 43.0 | 100 |
| XSum | OPT-2.7B | logrank | 91.61 | [87.94, 95.02] | 89.18 | 2.43 | 20.0 | 52.0 | 100 |
| XSum | OPT-2.7B | entropy | 45.96 | [41.09, 51.14] | 50.34 | -4.38 | 2.0 | 6.0 | 100 |
| XSum | OPT-2.7B | LRR | 91.88 | [87.93, 95.22] | 88.7 | 3.18 | 38.0 | 51.0 | 100 |
