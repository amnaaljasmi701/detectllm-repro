import re
import numpy as np
from repro import perturb as P

TEXT = " ".join(f"w{i}" for i in range(60))


def fake_filler(masked):
    out = []
    for t in masked:
        n = len(re.findall(r"<extra_id_\d+>", t))
        out.append("<pad> " + " ".join(f"<extra_id_{i}> FILL{i}" for i in range(n)) + f" <extra_id_{n}></s>")
    return out


def test_mask_count_and_buffer():
    rng = np.random.default_rng(0)
    m = P.tokenize_and_mask(TEXT, span_length=2, pct=0.3, rng=rng)
    n = len(re.findall(r"<extra_id_\d+>", m))
    assert n == int(0.3 * 60 / (2 + 2))
    toks = m.split(" ")
    idx = [i for i, t in enumerate(toks) if t.startswith("<extra_id_")]
    assert all(b - a >= 2 for a, b in zip(idx, idx[1:]))     # buffer of >= 1 word between masks


def test_perturb_round_trip():
    out = P.perturb_texts([TEXT] * 5, fake_filler, span_length=2, pct=0.3, seed=0)
    assert len(out) == 5
    for o in out:
        assert "<extra_id" not in o and "FILL0" in o
        assert o != TEXT


def test_failed_fills_are_retried_then_empty():
    calls = {"n": 0}

    def bad_filler(masked):
        calls["n"] += 1
        return ["<pad> nothing</s>" for _ in masked]
    out = P.perturb_texts([TEXT], bad_filler, max_attempts=3)
    assert out == [""] and calls["n"] == 4


def test_seed_reproducible():
    a = P.perturb_texts([TEXT] * 3, fake_filler, seed=7)
    b = P.perturb_texts([TEXT] * 3, fake_filler, seed=7)
    assert a == b
