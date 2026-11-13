"""T5 span-masking perturbations (ported from the original ``baselines/detectGPT.py``).

Changes: an explicit ``numpy.random.Generator`` replaces the global RNG (reproducible across
runs/seeds); a guard avoids an infinite loop on very short texts; the filler is injected,
so the logic can be unit-tested with a fake filler (tests/test_perturb.py).
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

PATTERN = re.compile(r"<extra_id_\d+>")
MASK = "<<<mask>>>"


def tokenize_and_mask(text: str, span_length: int, pct: float, rng: np.random.Generator,
                      buffer_size: int = 1, ceil_pct: bool = False) -> str:
    tokens = text.split(" ")
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)
    if len(tokens) <= span_length + 2 * buffer_size:
        n_spans = 0
    n_masks, guard = 0, 0
    while n_masks < n_spans and guard < 10_000:
        guard += 1
        start = int(rng.integers(0, len(tokens) - span_length))
        end = start + span_length
        s0, s1 = max(0, start - buffer_size), min(len(tokens), end + buffer_size)
        if MASK not in tokens[s0:s1]:
            tokens[start:end] = [MASK]
            n_masks += 1
    k = 0
    for i, t in enumerate(tokens):
        if t == MASK:
            tokens[i] = f"<extra_id_{k}>"
            k += 1
    assert k == n_masks
    return " ".join(tokens)


def count_masks(texts: list[str]) -> list[int]:
    return [len([x for x in t.split() if x.startswith("<extra_id_")]) for t in texts]


def extract_fills(raw: list[str]) -> list[list[str]]:
    raw = [x.replace("<pad>", "").replace("</s>", "").strip() for x in raw]
    return [[y.strip() for y in PATTERN.split(x)[1:-1]] for x in raw]


def apply_extracted_fills(masked: list[str], fills: list[list[str]]) -> list[str]:
    tokens = [x.split(" ") for x in masked]
    n_exp = count_masks(masked)
    for idx, (toks, f, n) in enumerate(zip(tokens, fills, n_exp)):
        if len(f) < n:
            tokens[idx] = []          # signals failure -> retried by the caller
        else:
            for j in range(n):
                toks[toks.index(f"<extra_id_{j}>")] = f[j]
    return [" ".join(t) for t in tokens]


def perturb_texts(texts: list[str], fill_fn: Callable[[list[str]], list[str]], span_length: int = 2,
                  pct: float = 0.3, chunk_size: int = 20, seed: int = 0, max_attempts: int = 10) -> list[str]:
    """Return one perturbed version for every input text (texts may repeat for k perturbations)."""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for c in range(0, len(texts), chunk_size):
        chunk = texts[c:c + chunk_size]
        masked = [tokenize_and_mask(x, span_length, pct, rng) for x in chunk]
        pert = apply_extracted_fills(masked, extract_fills(fill_fn(masked)))
        attempts = 1
        while "" in pert and attempts <= max_attempts:
            idxs = [i for i, x in enumerate(pert) if x == ""]
            masked2 = [tokenize_and_mask(chunk[i], span_length, pct, rng) for i in idxs]
            new = apply_extracted_fills(masked2, extract_fills(fill_fn(masked2)))
            for i, x in zip(idxs, new):
                pert[i] = x
            attempts += 1
        # texts that still failed fall back to NaN downstream (scored as empty)
        out.extend(pert)
    return out
