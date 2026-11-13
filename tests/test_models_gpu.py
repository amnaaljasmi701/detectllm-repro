"""Runs on Colab (needs torch + transformers + internet). Skipped automatically elsewhere.

Checks that the batched GPU implementation reproduces the original one-text-at-a-time code.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def lm():
    from repro.models import CausalLM
    try:
        return CausalLM("gpt2", dtype="fp32")
    except Exception as e:  # no internet
        pytest.skip(f"cannot load gpt2: {e}")


TEXTS = ["The quick brown fox jumps over the lazy dog. " * 3,
         "In 1998, the committee published a report on regional water policy and its effects on farmers.",
         "Hello world"]


def original_ll(lm, text):
    enc = lm.tok(text, return_tensors="pt").to(lm.device)
    return -lm.model(**enc, labels=enc["input_ids"]).loss.item()


def original_rank(lm, text, log):
    enc = lm.tok(text, return_tensors="pt").to(lm.device)
    logits = lm.model(**enc).logits[:, :-1]
    labels = enc["input_ids"][:, 1:]
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1].float() + 1
    if log:
        ranks = torch.log(ranks)
    return ranks.mean().item()


@torch.no_grad()
def test_batched_stats_equal_original(lm):
    st = lm.token_stats(TEXTS, batch_size=3)
    for t, s in zip(TEXTS, st):
        assert abs(s["ll"] - original_ll(lm, t)) < 1e-3
        # padding changes logits by ~1e-6, which can swap two nearly tied low-probability tokens,
        # so ranks are compared with a small tolerance
        assert abs(s["logrank"] - original_rank(lm, t, True)) < 1e-2
        assert abs(s["rank"] - original_rank(lm, t, False)) <= 0.01 * original_rank(lm, t, False) + 1e-6


def test_sample_keeps_prompt_and_length(lm):
    human = [" ".join(["The city council met on Monday to discuss the new budget."] * 30)] * 2
    out = lm.sample(human, prompt_tokens=30, max_length=80, min_length=60, min_words=20, batch_size=2)
    prompt = lm.tok.decode(lm.tok(human[0])["input_ids"][:30])
    assert all(o.startswith(prompt) for o in out)
    assert all(len(o.split()) >= 20 for o in out)
