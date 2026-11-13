"""Evaluation metrics. Label 1 = machine-generated (positive), 0 = human."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def _arrays(human, machine):
    h = np.asarray(human, dtype=float)
    m = np.asarray(machine, dtype=float)
    # +inf/-inf are legitimate extreme scores; NaN is not -> drop pairwise
    keep = ~(np.isnan(h) | np.isnan(m)) if len(h) == len(m) else None
    if keep is not None:
        h, m = h[keep], m[keep]
    else:
        h, m = h[~np.isnan(h)], m[~np.isnan(m)]
    # replace infinities with finite extremes so sklearn accepts them
    finite = np.concatenate([h[np.isfinite(h)], m[np.isfinite(m)]])
    big = (np.abs(finite).max() if finite.size else 1.0) * 10 + 1
    h = np.clip(np.nan_to_num(h, posinf=big, neginf=-big), -big, big)
    m = np.clip(np.nan_to_num(m, posinf=big, neginf=-big), -big, big)
    return h, m


def auroc(human, machine) -> float:
    h, m = _arrays(human, machine)
    y = np.r_[np.zeros(len(h)), np.ones(len(m))]
    return float(roc_auc_score(y, np.r_[h, m]))


def pr_auc(human, machine) -> float:
    h, m = _arrays(human, machine)
    y = np.r_[np.zeros(len(h)), np.ones(len(m))]
    p, r, _ = precision_recall_curve(y, np.r_[h, m])
    return float(auc(r, p))


def tpr_at_fpr(human, machine, fpr: float = 0.01) -> float:
    """Share of machine texts detected when the threshold lets through at most ``fpr`` of human texts.

    The threshold is the smallest value such that the fraction of human scores strictly above
    it is <= fpr (conservative: never exceeds the target FPR on this sample).
    """
    h, m = _arrays(human, machine)
    hs = np.sort(h)[::-1]
    k = int(np.floor(fpr * len(hs)))          # number of human texts allowed above threshold
    thr = hs[k] if k < len(hs) else -np.inf   # scores must be > thr
    return float((m > thr).mean())


def best_accuracy(human, machine) -> float:
    """Accuracy at the best single threshold (oracle, reported for intuition only)."""
    h, m = _arrays(human, machine)
    y = np.r_[np.zeros(len(h)), np.ones(len(m))]
    s = np.r_[h, m]
    fpr, tpr, _ = roc_curve(y, s)
    acc = (tpr * len(m) + (1 - fpr) * len(h)) / (len(h) + len(m))
    return float(acc.max())


def bootstrap_auroc_ci(human, machine, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap (resample text pairs) percentile CI for AUROC."""
    h, m = _arrays(human, machine)
    n = min(len(h), len(m))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        hb, mb = h[idx], m[idx]
        if np.all(np.r_[hb, mb] == hb[0]):
            continue
        vals.append(auroc(hb, mb))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(h1, m1, h2, m2, n_boot: int = 1000, seed: int = 0):
    """AUROC(method1) - AUROC(method2) on the same texts, with a 95% CI and a one-sided p-value
    for the hypothesis that method1 is not better (fraction of bootstrap diffs <= 0)."""
    h1, m1 = _arrays(h1, m1)
    h2, m2 = _arrays(h2, m2)
    n = min(len(h1), len(m1), len(h2), len(m2))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d.append(auroc(h1[idx], m1[idx]) - auroc(h2[idx], m2[idx]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(auroc(h1[:n], m1[:n]) - auroc(h2[:n], m2[:n])), float(lo), float(hi), float((d <= 0).mean())


def all_metrics(human, machine, ci: bool = True, n_boot: int = 1000) -> dict:
    out = {
        "auroc": auroc(human, machine),
        "pr_auc": pr_auc(human, machine),
        "tpr@1%fpr": tpr_at_fpr(human, machine, 0.01),
        "tpr@5%fpr": tpr_at_fpr(human, machine, 0.05),
        "best_acc": best_accuracy(human, machine),
        "n": int(min(len(human), len(machine))),
    }
    if ci:
        out["auroc_lo"], out["auroc_hi"] = bootstrap_auroc_ci(human, machine, n_boot=n_boot)
    return out
