"""Reproduction of DetectLLM (Su et al., Findings of EMNLP 2023): LRR and NPR zero-shot detectors.

Modules
-------
stats      : token-level statistics (log-prob, rank, entropy) from logits  (numpy reference implementation)
methods    : detection criteria (likelihood, rank, log-rank, entropy, LRR, DetectGPT, NPR)
metrics    : AUROC, PR-AUC, TPR at fixed FPR, bootstrap confidence intervals
data       : loading and filtering human-written texts (XSum, SQuAD, WritingPrompts, Arabic Wikipedia)
models     : HuggingFace causal-LM wrapper (scoring + sampling) used on GPU
perturb    : T5 span-masking perturbations (DetectGPT / NPR)
pipeline   : end-to-end experiment runner with caching
experiments: experiment definitions and presets (smoke / quick / full)
"""
__version__ = "1.0.0"
