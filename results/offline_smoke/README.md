CPU-only end-to-end check (tests/smoke_offline.py): GPT-2 small (ONNX), 44 SQuAD v1.1 *dev*
paragraphs (all >250-word paragraphs passing the 512-token filter), n=44 pairs, top-k 50, seed 0.
This is a pipeline check, not one of the reproduction runs (different model, split and size).

| method | AUROC | 95% CI | TPR@1%FPR | AUROC, texts cut to 50 words |
|---|---|---|---|---|
| likelihood | 92.0 | [87.0, 97.0] | 20.5 | 67.3 |
| rank | 91.6 | [86.8, 96.0] | 27.3 | 65.3 |
| log-rank | 96.7 | [92.8, 99.4] | 31.8 | 71.9 |
| entropy | 44.8 | [36.4, 51.9] | 0.0 | 56.0 |
| **LRR** | **99.4** | [98.1, 100.0] | **75.0** | **79.9** |

Generation took 17.5 min on 2 CPU cores (no KV cache); scoring 47 s.
