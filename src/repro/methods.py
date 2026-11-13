"""Detection criteria.

Every function returns a score where **higher means "more likely machine-generated"**,
matching the sign conventions of the original repository (``baselines/all_baselines.py``):

=============  ===============================================  ==========================
criterion      formula (per text)                               original code
=============  ===============================================  ==========================
likelihood     mean log p                                       ``res['ll']``
rank           -(mean rank)                                     ``-get_rank(log=False)``
logrank        -(mean log rank)                                 ``-res['logrank']``
entropy        mean entropy                                     ``get_entropy``
LRR            -(mean log p) / (mean log rank)                  ``-ll / logrank``
DetectGPT      (ll - mean ll_pert) / std ll_pert                ``(ll - p_ll) / p_ll_std``
NPR            mean logrank_pert / logrank                      ``p_logrank / logrank``
=============  ===============================================  ==========================
"""
from __future__ import annotations

import math
import numpy as np

PERTURBATION_FREE = ["likelihood", "rank", "logrank", "entropy", "LRR"]
PERTURBATION_BASED = ["DetectGPT", "NPR"]


def likelihood(s: dict) -> float:
    return s["ll"]


def rank(s: dict) -> float:
    return -s["rank"]


def logrank(s: dict) -> float:
    return -s["logrank"]


def entropy(s: dict) -> float:
    return s["entropy"]


def lrr(s: dict) -> float:
    """Log-Likelihood Log-Rank Ratio (Eq. 1 of the paper)."""
    if s["logrank"] == 0:          # every token was the arg-max: degenerate, maximally "machine"
        return float("inf")
    return -s["ll"] / s["logrank"]


def _clean(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def detectgpt(s: dict, pert_lls: list[float], n: int) -> float:
    """Normalised DetectGPT score with the first ``n`` perturbations (as in the original code)."""
    p = _clean(pert_lls[:n])
    if not p:
        return float("nan")
    mu = float(np.mean(p))
    sd = float(np.std(p)) if len(p) > 1 else 1.0
    if sd == 0:
        sd = 1.0
    return (s["ll"] - mu) / sd


def npr(s: dict, pert_logranks: list[float], n: int) -> float:
    """Normalised Log-Rank Perturbation (Eq. 2 of the paper)."""
    p = _clean(pert_logranks[:n])
    if not p or s["logrank"] == 0:
        return float("nan")
    return float(np.mean(p)) / s["logrank"]


FREE_FNS = {"likelihood": likelihood, "rank": rank, "logrank": logrank, "entropy": entropy, "LRR": lrr}
