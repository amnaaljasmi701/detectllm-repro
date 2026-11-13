import math
import numpy as np
from repro import methods as M


def test_signs_higher_is_machine():
    human = {"ll": -3.5, "logrank": 1.8, "rank": 40.0, "entropy": 3.0}
    machine = {"ll": -2.0, "logrank": 0.9, "rank": 8.0, "entropy": 2.5}
    for name in ["likelihood", "rank", "logrank", "LRR"]:
        assert M.FREE_FNS[name](machine) > M.FREE_FNS[name](human), name


def test_npr_and_detectgpt():
    s = {"ll": -2.0, "logrank": 1.0}
    assert np.isclose(M.npr(s, [1.2, 1.4, float("nan")], 3), 1.3)
    assert np.isclose(M.npr(s, [1.2, 1.4, 9.0], 2), 1.3)          # only the first n perturbations
    d = M.detectgpt(s, [-2.5, -3.5], 2)
    assert np.isclose(d, (-2.0 - (-3.0)) / 0.5)
    assert np.isclose(M.detectgpt(s, [-2.5], 1), 0.5)              # std := 1 with one perturbation
    assert math.isnan(M.npr(s, [], 5))
