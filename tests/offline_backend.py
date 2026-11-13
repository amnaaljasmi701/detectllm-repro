"""Offline CPU backend (GPT-2 small exported to ONNX + the original OpenAI BPE files).

Used only to smoke-test the full pipeline without a GPU or HuggingFace access
(``python tests/smoke_offline.py``). It exposes the same interface as ``repro.models.CausalLM``
and computes statistics with the numpy reference implementation in ``repro.stats``.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np

from repro import stats as ST


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class GPT2BPE:
    """OpenAI's GPT-2 encoder.py, with ``re`` instead of ``regex`` (letters approximated by [^\\W\\d_])."""

    def __init__(self, encoder_json, vocab_bpe):
        self.encoder = json.load(open(encoder_json))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(vocab_bpe, encoding="utf-8").read().split("\n")[1:-1]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        L = r"[^\W\d_]"
        self.pat = re.compile(rf"""'s|'t|'re|'ve|'m|'ll|'d| ?{L}+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        self.cache = {}
        self.eos_token_id = 50256

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        while pairs:
            big = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if big not in self.bpe_ranks:
                break
            first, second = big
            new, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new.extend(word[i:])
                    break
                new.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        self.cache[token] = word
        return word

    def encode(self, text):
        ids = []
        for tok in self.pat.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(tok))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[i] for i in ids if i != self.eos_token_id)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")


class OnnxGPT2:
    name = "gpt2-onnx"

    def __init__(self, onnx_path, encoder_json, vocab_bpe, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.tok = GPT2BPE(encoder_json, vocab_bpe)
        self.eos_id = 50256

    def free(self):
        pass

    def _logits(self, ids_2d):
        x = np.asarray(ids_2d, dtype=np.int64)[:, None, :]          # [B, 1, T]
        return self.sess.run(["output1"], {self.inp: x})[0][:, 0]    # [B, T, V]

    def n_tokens(self, texts):
        return [len(self.tok.encode(t)) for t in texts]

    def token_stats(self, texts, batch_size=None, return_token_level=False):
        out = []
        for t in texts:
            ids = self.tok.encode(t)[:1024]
            if len(ids) < 2:
                out.append({"ll": float("nan"), "logrank": float("nan"), "rank": float("nan"),
                            "entropy": float("nan"), "n_tokens": 0})
                continue
            logits = self._logits([ids])[0][:-1]
            out.append(ST.text_stats_from_logits(logits, np.asarray(ids[1:])))
        return out

    def sample(self, texts, prompt_tokens=30, max_length=200, min_length=150, min_words=55, temperature=1.0,
               top_k=50, top_p=1.0, batch_size=25, seed=42, max_tries=4):
        rng = np.random.default_rng(seed)
        outs = []
        for s in range(0, len(texts), batch_size):
            batch = [self.tok.encode(t)[:prompt_tokens] for t in texts[s:s + batch_size]]
            ids = np.array(batch)                                          # all prompts have 30 tokens
            done = np.zeros(len(batch), bool)
            while ids.shape[1] < max_length:
                logits = self._logits(ids)[:, -1].astype(np.float64) / temperature
                if ids.shape[1] < min_length:
                    logits[:, self.eos_id] = -np.inf
                if top_k and top_k > 0:
                    kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                    logits = np.where(logits < kth, -np.inf, logits)
                p = np.exp(logits - logits.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                nxt = np.array([rng.choice(len(pi), p=pi) for pi in p])
                nxt = np.where(done, self.eos_id, nxt)
                done |= nxt == self.eos_id
                ids = np.concatenate([ids, nxt[:, None]], axis=1)
                if done.all():
                    break
            outs.extend(self.tok.decode([i for i in row]) for row in ids.tolist())
        return outs
