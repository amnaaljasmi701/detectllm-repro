from repro import data as D


def test_trim_to_shorter_length():
    a, b = D.trim_to_shorter_length("a b c d e", "x y z")
    assert a == "a b c" and b == "x y z"


def test_prepare_pool_filters_and_is_deterministic():
    long = [" ".join(["w"] * 260) + f" {i}" for i in range(30)]
    short = ["short text"] * 5
    texts = long + short + [long[0]]
    p1 = D.prepare_pool(texts, "xsum", seed=0, token_len_fn=lambda xs: [len(x.split()) for x in xs], max_tokens=1000)
    p2 = D.prepare_pool(texts, "xsum", seed=0, token_len_fn=lambda xs: [len(x.split()) for x in xs], max_tokens=1000)
    assert p1 == p2 and len(p1) == 30            # duplicate and short texts removed
    p3 = D.prepare_pool(texts, "xsum", seed=1)
    assert sorted(p3) == sorted(p1) and p3 != p1  # other seed -> other order
    p4 = D.prepare_pool(texts, "xsum", seed=0, token_len_fn=lambda xs: [len(x.split()) for x in xs], max_tokens=200)
    assert p4 == []


def test_newlines_collapsed():
    p = D.prepare_pool(["a\n\nb " + " ".join(["w"] * 300)], "squad")
    assert "\n" not in p[0]
