## Claim 1: LRR beats every other perturbation-free method

| dataset | model | LRR ours | best other ours | gain ours | LRR paper | best other paper | gain paper | LRR is best (ours) |
|---|---|---|---|---|---|---|---|---|
| SQuAD | GPT2-xl | 97.83 | 95.12 | 2.71 | 97.42 | 94.33 | 3.09 | True |
| SQuAD | Neo-2.7B | 97.23 | 90.68 | 6.55 | 95.74 | 89.52 | 6.22 | True |
| SQuAD | OPT-2.7B | 95.62 | 91.47 | 4.15 | 95.89 | 91.76 | 4.13 | True |
| WritingPrompts | GPT2-xl | 98.71 | 97.89 | 0.82 | 98.34 | 98.02 | 0.32 | True |
| WritingPrompts | Neo-2.7B | 97.38 | 96.29 | 1.09 | 98.02 | 97.15 | 0.87 | True |
| WritingPrompts | OPT-2.7B | 97.32 | 94.76 | 2.56 | 96.45 | 96.32 | 0.13 | True |
| XSum | GPT2-xl | 92.74 | 92.08 | 0.66 | 93.47 | 91.75 | 1.72 | True |
| XSum | Neo-2.7B | 95.27 | 91.24 | 4.03 | 92.24 | 90.79 | 1.45 | True |
| XSum | OPT-2.7B | 91.88 | 91.61 | 0.27 | 88.7 | 89.18 | -0.48 | True |

Mean gain ours: 2.54 | mean gain paper (same 3 models): 1.94 | paper's 7-model claim: 3.9
LRR best in 9/9 settings (ours)
