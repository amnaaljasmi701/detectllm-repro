"""The fast statistics must equal a line-by-line numpy port of the original repository's code."""
import numpy as np
from repro import stats, methods


def original_rank(logits, labels, log=False):
    # port of baselines/rank.py::get_rank (argsort over the vocabulary, 1-indexed)
    order = np.argsort(-logits, axis=-1, kind="stable")
    ranks = np.array([np.where(order[t] == labels[t])[0][0] for t in range(len(labels))]) + 1.0
    return float(np.log(ranks).mean()) if log else float(ranks.mean())


def original_ll(logits, labels):
    # HF causal-LM loss = mean token negative log-likelihood; get_ll returns -loss
    lp = logits - logits.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    return float(lp[np.arange(len(labels)), labels].mean())


def test_rank_matches_argsort():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T, V = rng.integers(5, 40), rng.integers(50, 400)
        logits = rng.normal(size=(T, V)) * 3
        labels = rng.integers(0, V, T)
        s = stats.text_stats_from_logits(logits, labels)
        assert np.isclose(s["rank"], original_rank(logits, labels))
        assert np.isclose(s["logrank"], original_rank(logits, labels, log=True))
        assert np.isclose(s["ll"], original_ll(logits, labels))


def test_rank_one_for_argmax_tokens():
    logits = np.array([[0., 5., 1.], [3., 0., 1.]])
    labels = np.array([1, 0])
    s = stats.text_stats_from_logits(logits, labels)
    assert s["rank"] == 1 and s["logrank"] == 0
    assert methods.lrr(s) == float("inf")      # degenerate case handled, no ZeroDivisionError


def test_entropy_uniform():
    V = 64
    s = stats.text_stats_from_logits(np.zeros((3, V)), np.array([0, 1, 2]))
    assert np.isclose(s["entropy"], np.log(V))


def test_lrr_equals_original_expression():
    s = {"ll": -2.7, "logrank": 1.3}
    assert np.isclose(methods.lrr(s), -s["ll"] / s["logrank"])
    # paper Eq.1: absolute ratio of the two means
    assert np.isclose(methods.lrr(s), abs(s["ll"] / s["logrank"]))
