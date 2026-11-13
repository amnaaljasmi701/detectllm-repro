## LRR minus best other perturbation-free criterion (paired bootstrap, 2000 resamples)

| exp | dataset | generator | seed | best other | LRR - best x100 | 95% CI | P(LRR not better) | significant |
|---|---|---|---|---|---|---|---|---|
| E1 | SQuAD | GPT2-xl | 0 | logrank | 2.71 | [0.64, 5.04] | 0.005 | True |
| E1 | WritingPrompts | GPT2-xl | 0 | logrank | 0.82 | [-0.08, 1.89] | 0.038 | False |
| E1 | XSum | GPT2-xl | 0 | logrank | 0.66 | [-1.77, 2.95] | 0.281 | False |
| E1 | SQuAD | Neo-2.7B | 0 | logrank | 6.55 | [3.97, 9.51] | 0.0 | True |
| E1 | WritingPrompts | Neo-2.7B | 0 | logrank | 1.09 | [-0.27, 2.90] | 0.069 | False |
| E1 | XSum | Neo-2.7B | 0 | logrank | 4.03 | [1.72, 6.39] | 0.0 | True |
| E1 | SQuAD | OPT-2.7B | 0 | logrank | 4.15 | [1.73, 6.79] | 0.0 | True |
| E1 | WritingPrompts | OPT-2.7B | 0 | logrank | 2.56 | [0.94, 4.54] | 0.001 | True |
| E1 | XSum | OPT-2.7B | 0 | logrank | 0.27 | [-2.48, 2.98] | 0.428 | False |
| E3 | XSum | GPT2-xl | 1 | logrank | 3.03 | [0.35, 5.90] | 0.017 | True |
| E3 | XSum | GPT2-xl | 2 | logrank | 1.97 | [-0.63, 4.44] | 0.076 | False |
| E5-arabic | Arabic Wikipedia | Qwen2.5-1.5B | 0 | rank | -0.11 | [-5.15, 4.44] | 0.55 | False |
| E5-arabic | Arabic Wikipedia | AraGPT2-M | 0 | logrank | 0.34 | [-0.45, 1.52] | 0.289 | False |
| E5-control | XSum | Qwen2.5-1.5B | 0 | logrank | 5.64 | [2.24, 9.27] | 0.0 | True |
