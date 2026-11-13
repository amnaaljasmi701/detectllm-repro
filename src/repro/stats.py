"""Token-level statistics computed from next-token logits.

This is the framework-independent (numpy) reference implementation. The GPU code in
``models.py`` computes exactly the same quantities with torch; ``tests/test_stats.py``
checks both against a direct re-implementation of the original repository's code.

Conventions (identical to the original DetectLLM / DetectGPT code):
* ``logits[t]`` is the model's distribution for token ``t+1``; labels are ``ids[1:]``.
* rank is 1-indexed: the most likely token has rank 1.
* per-text statistics are means over the scored tokens.
"""
from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def token_logprobs(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """log p(x_t | x_<t) for each position. logits: [T, V], labels: [T]."""
    lp = log_softmax(logits.astype(np.float64))
    return lp[np.arange(len(labels)), labels]


def token_ranks(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1-indexed rank of the observed token among all vocabulary items.

    The original code sorts the full vocabulary (``argsort``) and finds the label's
    position.  Counting the logits that are strictly larger than the label's logit gives
    the same rank whenever there are no exact ties, and is O(V) instead of O(V log V).
    With ties, this returns the *best* rank among the tied tokens (argsort order is
    arbitrary there); with float logits ties are extremely rare (tested in test_stats).
    """
    label_logit = logits[np.arange(len(labels)), labels]
    return (logits > label_logit[:, None]).sum(axis=-1) + 1


def token_entropy(logits: np.ndarray) -> np.ndarray:
    lp = log_softmax(logits.astype(np.float64))
    return -(np.exp(lp) * lp).sum(axis=-1)


def text_stats_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Aggregate statistics for one text (all positions are valid)."""
    lp = token_logprobs(logits, labels)
    rk = token_ranks(logits, labels).astype(np.float64)
    ent = token_entropy(logits)
    return {
        "ll": float(lp.mean()),                 # mean log-likelihood (negative number)
        "logrank": float(np.log(rk).mean()),    # mean log-rank (>= 0)
        "rank": float(rk.mean()),               # mean rank (>= 1)
        "entropy": float(ent.mean()),           # mean predictive entropy
        "n_tokens": int(len(labels)),
    }
