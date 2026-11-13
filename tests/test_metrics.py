import numpy as np
from sklearn.metrics import roc_auc_score
from repro import metrics as MET


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(1)
    h, m = rng.normal(0, 1, 200), rng.normal(0.8, 1, 200)
    ref = roc_auc_score(np.r_[np.zeros(200), np.ones(200)], np.r_[h, m])
    assert np.isclose(MET.auroc(h, m), ref)


def test_perfect_and_inverted():
    assert MET.auroc([0, 1, 2], [3, 4, 5]) == 1.0
    assert MET.auroc([3, 4, 5], [0, 1, 2]) == 0.0


def test_infinite_scores_are_handled():
    assert MET.auroc([0.1, 0.2], [float("inf"), 0.3]) == 1.0


def test_tpr_at_fpr():
    human = np.arange(100)                # 0..99
    machine = np.array([98.5, 99.5, 50, 10])
    # 1% FPR -> one human text (99) may be above the threshold -> threshold 98 -> 98.5 and 99.5 detected
    assert MET.tpr_at_fpr(human, machine, 0.01) == 0.5
    assert MET.tpr_at_fpr(human, machine, 0.0) == 0.25    # only 99.5 > 99


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(2)
    h, m = rng.normal(0, 1, 150), rng.normal(1, 1, 150)
    lo, hi = MET.bootstrap_auroc_ci(h, m, n_boot=300)
    assert lo <= MET.auroc(h, m) <= hi


def test_paired_diff_sign():
    rng = np.random.default_rng(3)
    h, m = rng.normal(0, 1, 150), rng.normal(1.5, 1, 150)
    d, lo, hi, p = MET.paired_bootstrap_diff(h, m, h + rng.normal(0, 2, 150), m + rng.normal(0, 2, 150), n_boot=300)
    assert d > 0 and p < 0.05
