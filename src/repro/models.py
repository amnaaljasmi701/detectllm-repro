"""HuggingFace causal-LM wrapper: batched token statistics and sampling.

Differences from the original code (all documented in docs/AUDIT.md and the report):
* scoring is batched with right padding + attention mask instead of one text at a time
  (numerically equivalent; checked by ``tests/test_models_gpu.py`` on Colab);
* ranks are computed by counting larger logits instead of a full ``argsort`` (identical
  except for exact ties);
* the sampling call passes ``top_k`` explicitly. The original passed nothing, which in
  HuggingFace ``generate`` silently means ``top_k=50``; we keep 50 as the default so the
  reproduction matches the *code* and expose ``top_k=0`` for true pure sampling.
"""
from __future__ import annotations

import gc
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # allows importing the package (and running numpy tests) without torch
    torch = None


def _dtype(name: str, dtype: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # "auto": the original used fp32 for <= 2.7B models; on a 16 GB T4 we keep fp32 for
    # gpt2-xl and use fp16 for the 2.7B models to fit memory (see report, Section 3).
    # Qwen2.5 is numerically fragile in fp16, and 1.5B fits in fp32 on a T4, so it stays fp32.
    big = any(k in name.lower() for k in ["2.7b", "6b", "13b", "20b"])
    return torch.float16 if big else torch.float32


class CausalLM:
    def __init__(self, name: str, device: Optional[str] = None, dtype: str = "auto", cache_dir=None):
        import transformers
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # NB: the original passed `fast=False` for OPT, which transformers ignores, so the fast
        # tokenizer was effectively used. We keep that effective behaviour (docs/AUDIT.md, B1).
        tok_kw = {}
        self.tok = transformers.AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, **tok_kw)
        if self.tok.pad_token_id is None:
            if self.tok.eos_token is not None:
                self.tok.pad_token = self.tok.eos_token
            else:  # padded positions are masked out, so any id works
                self.tok.pad_token = self.tok.convert_ids_to_tokens(0)
        self.tok.padding_side = "right"
        self.torch_dtype = _dtype(name, dtype) if self.device != "cpu" else torch.float32
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, cache_dir=cache_dir).to(self.device).eval()
        self.vocab = self.model.get_output_embeddings().weight.shape[0]
        self.eos_id = self.tok.eos_token_id

    # ------------------------------------------------------------------ utilities
    def n_tokens(self, texts: list[str]) -> list[int]:
        return [len(x) for x in self.tok(texts)["input_ids"]]

    def free(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _auto_bs(self, max_len: int) -> int:
        # three [B, T, V] float32 tensors are alive at once (logits, log-softmax, probs)
        budget = 2.5e9 if self.device != "cpu" else 1e9
        return max(1, int(budget // (3 * 4 * max(max_len, 1) * self.vocab)))

    # ------------------------------------------------------------------ scoring
    @torch.no_grad() if torch else (lambda f: f)
    def token_stats(self, texts: list[str], batch_size: Optional[int] = None,
                    return_token_level: bool = False) -> list[dict]:
        """Mean log-prob, log-rank, rank and entropy for each text (see stats.py)."""
        out: list[dict] = [None] * len(texts)
        lens = self.n_tokens(texts)
        order = sorted(range(len(texts)), key=lambda i: lens[i])  # length-bucketing, less padding
        i = 0
        while i < len(order):
            if batch_size:
                bs = batch_size
            else:
                bs0 = self._auto_bs(lens[order[i]])
                bs = self._auto_bs(lens[order[min(i + bs0 - 1, len(order) - 1)]])
            idx = order[i:i + bs]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**enc).logits[:, :-1].float()
            labels = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            logp_all = F.log_softmax(logits, dim=-1)
            lp = logp_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            label_logit = logits.gather(-1, labels.unsqueeze(-1))
            rk = (logits > label_logit).sum(-1).float() + 1.0
            ent = -(logp_all.exp() * logp_all).sum(-1)
            for b, j in enumerate(idx):
                m = mask[b]
                n = int(m.sum())
                if n == 0:
                    out[j] = {"ll": float("nan"), "logrank": float("nan"), "rank": float("nan"),
                              "entropy": float("nan"), "n_tokens": 0}
                    continue
                if not torch.isfinite(lp[b][m]).all():
                    print(f"[warn] non-finite log-probs for {self.name} ({self.torch_dtype}); consider dtype=fp32")
                d = {
                    "ll": float(lp[b][m].mean()),
                    "logrank": float(torch.log(rk[b][m]).mean()),
                    "rank": float(rk[b][m].mean()),
                    "entropy": float(ent[b][m].mean()),
                    "n_tokens": n,
                }
                if return_token_level:
                    d["tok_logp"] = lp[b][m].cpu().numpy().tolist()
                    d["tok_rank"] = rk[b][m].cpu().numpy().astype(int).tolist()
                    d["tokens"] = self.tok.convert_ids_to_tokens(labels[b][m].cpu().tolist())
                out[j] = d
            del logits, logp_all, lp, rk, ent
            i += bs
        return out

    def lls(self, texts: list[str]) -> list[float]:
        return [s["ll"] for s in self.token_stats(texts)]

    # ------------------------------------------------------------------ sampling
    @torch.no_grad() if torch else (lambda f: f)
    def sample(self, texts: list[str], prompt_tokens: int = 30, max_length: int = 200, min_length: int = 150,
               min_words: int = 55, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0,
               batch_size: int = 25, seed: int = 42, max_tries: int = 4) -> list[str]:
        """Continue the first ``prompt_tokens`` tokens of each human text (original protocol).

        ``max_length``/``min_length`` count the prompt, as in the original call. The decoded
        output includes the prompt. A whole batch is re-sampled while any sample has fewer
        than ``min_words`` words (at most ``max_tries`` attempts), exactly like the original.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        outs: list[str] = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True)
            ids = enc["input_ids"][:, :prompt_tokens].to(self.device)
            am = enc["attention_mask"][:, :prompt_tokens].to(self.device)
            decoded = [""] * len(batch)
            tries = 0
            # Every prompt has exactly `prompt_tokens` tokens, so max/min_length (which count the
            # prompt) translate exactly into max/min_new_tokens. Passing *new* tokens explicitly
            # also overrides model-specific defaults (e.g. Qwen2.5 ships max_new_tokens=2048).
            plen = ids.shape[1]
            while min(len(x.split()) for x in decoded) < min_words:
                gen = self.model.generate(
                    input_ids=ids, attention_mask=am, do_sample=True, temperature=temperature,
                    top_k=top_k, top_p=top_p, max_new_tokens=max_length - plen,
                    min_new_tokens=max(0, min_length - plen),
                    pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_id,
                    repetition_penalty=1.0, num_beams=1)
                decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
                tries += 1
                if tries >= max_tries:
                    break
            outs.extend(decoded)
        return outs


class MaskFiller:
    """T5 span in-filling model used for DetectGPT / NPR perturbations."""

    def __init__(self, name: str = "t5-large", device: Optional[str] = None, dtype: str = "fp32", cache_dir=None):
        import transformers
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            name, torch_dtype=td, low_cpu_mem_usage=True, cache_dir=cache_dir).to(self.device).eval()
        n_pos = getattr(self.model.config, "n_positions", 512) or 512
        self.tok = transformers.AutoTokenizer.from_pretrained(name, model_max_length=n_pos, cache_dir=cache_dir)

    @torch.no_grad() if torch else (lambda f: f)
    def fill(self, masked_texts: list[str], top_p: float = 1.0) -> list[str]:
        """Sample span fills. top_k=50 reproduces HF's implicit default in the original call."""
        n_expected = [len([x for x in t.split() if x.startswith("<extra_id_")]) for t in masked_texts]
        stop_id = self.tok.encode(f"<extra_id_{max(n_expected)}>")[0]
        enc = self.tok(masked_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model.generate(**enc, max_length=150, do_sample=True, top_p=top_p, top_k=50,
                                  num_return_sequences=1, eos_token_id=stop_id)
        return self.tok.batch_decode(out, skip_special_tokens=False)

    def free(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
