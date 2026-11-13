"""End-to-end experiment runner with on-disk caching (safe to resume after a Colab disconnect).

Layout of ``out_dir``::

    data/<data_key>.json          human/machine pairs (+ generation metadata)
    stats/<stats_key>.json        per-text statistics under a scoring model
    pert/<pert_key>.json          perturbed texts (+ their statistics)
    scores/<run_id>.csv           per-text detector scores (for error analysis)
    results.jsonl                 one row per (run, method, n_perturbations)
    timing.jsonl                  wall-clock time of every stage (resource analysis)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import numpy as np

from . import data as D
from . import methods as M
from . import metrics as MET
from .perturb import perturb_texts


@dataclass
class RunCfg:
    exp: str
    dataset: str
    generator: str
    scorer: Optional[str] = None            # None = white-box setting (scorer == generator)
    n_samples: int = 300
    seed: int = 0                           # data shuffle seed (0 = original); sampling seed = 42 + seed
    temperature: float = 1.0
    top_k: int = 50                         # 50 = HF default silently used by the original code
    top_p: float = 1.0
    prompt_tokens: int = 30
    max_length: int = 200
    min_length: int = 150
    min_words: int = 55
    gen_batch: int = 25
    dtype: str = "auto"                     # generator precision
    scorer_dtype: Optional[str] = None      # defaults to dtype
    truncate_words: Optional[int] = None    # post-hoc truncation (length sensitivity)
    perturb: bool = False
    n_pert_texts: int = 100
    n_pert_list: tuple = (1, 5, 10, 20)
    mask_model: str = "t5-large"
    pct_words_masked: float = 0.3
    span_length: int = 2
    tag: str = ""

    @property
    def scorer_name(self) -> str:
        return self.scorer or self.generator

    def data_key(self) -> str:
        s = (f"{self.dataset}__{self.generator}__seed{self.seed}__n{self.n_samples}"
             f"__T{self.temperature}_k{self.top_k}_p{self.top_p}")
        return s.replace("/", "_")

    def stats_key(self) -> str:
        t = f"__trunc{self.truncate_words}" if self.truncate_words else ""
        return (f"{self.data_key()}__scorer={self.scorer_name}__{self.scorer_dtype or self.dtype}{t}").replace("/", "_")

    def run_id(self) -> str:
        h = hashlib.md5(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:8]
        return f"{self.exp}__{self.stats_key()}__{h}"


class Runner:
    """Keeps at most one causal LM in memory (Colab T4 has 15 GB) plus the mask filler."""

    def __init__(self, out_dir: str = "results", cache_dir: Optional[str] = None,
                 lm_factory: Optional[Callable] = None, filler_factory: Optional[Callable] = None,
                 token_len_fn_factory: Optional[Callable] = None, squad_local_json: Optional[str] = None,
                 n_boot: int = 1000):
        self.out = out_dir
        self.cache_dir = cache_dir
        for sub in ["data", "stats", "pert", "scores"]:
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        self._lm, self._lm_key = None, None
        self._filler, self._filler_name = None, None
        self.lm_factory = lm_factory or self._default_lm
        self.filler_factory = filler_factory or self._default_filler
        self.token_len_fn_factory = token_len_fn_factory or self._default_len_fn
        self.squad_local_json = squad_local_json
        self.n_boot = n_boot
        self._pools: dict = {}
        self.env = self._env()

    # ------------------------------------------------------------------ resources
    @staticmethod
    def _env() -> dict:
        env = {"python": platform.python_version(), "machine": platform.machine()}
        try:
            import torch
            env["torch"] = torch.__version__
            env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            import transformers
            env["transformers"] = transformers.__version__
        except Exception:  # noqa: BLE001
            env["gpu"] = "cpu (no torch)"
        return env

    def _default_lm(self, name, dtype):
        from .models import CausalLM
        return CausalLM(name, dtype=dtype, cache_dir=self.cache_dir)

    def _default_filler(self, name):
        from .models import MaskFiller
        return MaskFiller(name, cache_dir=self.cache_dir)

    def _default_len_fn(self, dataset: str, lm):
        if dataset == "arwiki":
            # T5's English vocabulary cannot tokenise Arabic; a model-independent word count keeps
            # the *same* human texts for every Arabic model (needed for cross-model comparisons).
            return lambda texts: [len(t.split()) for t in texts]
        import transformers
        t5 = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=512, cache_dir=self.cache_dir)
        return lambda texts: [len(x) for x in t5(texts)["input_ids"]]

    def lm(self, name: str, dtype: str):
        key = (name, dtype)
        if self._lm_key != key:
            if self._lm is not None:
                self._lm.free()
                self._lm = None
            t0 = time.time()
            self._lm = self.lm_factory(name, dtype)
            self._lm_key = key
            self._log_time("load_model", name, time.time() - t0)
        return self._lm

    def filler(self, name: str):
        if self._filler_name != name:
            if self._filler is not None:
                self._filler.free()
            t0 = time.time()
            self._filler = self.filler_factory(name)
            self._filler_name = name
            self._log_time("load_mask_model", name, time.time() - t0)
        return self._filler

    def _log_time(self, stage: str, what: str, sec: float, **kw):
        with open(os.path.join(self.out, "timing.jsonl"), "a") as f:
            f.write(json.dumps({"stage": stage, "what": what, "seconds": round(sec, 2), **kw, **self.env}) + "\n")

    # ------------------------------------------------------------------ stages
    def get_pairs(self, cfg: RunCfg) -> dict:
        path = os.path.join(self.out, "data", cfg.data_key() + ".json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        lm = self.lm(cfg.generator, cfg.dtype)
        pool_key = (cfg.dataset, cfg.seed)
        if pool_key not in self._pools:
            t0 = time.time()
            len_fn = self.token_len_fn_factory(cfg.dataset, lm)
            kw = {"local_json": self.squad_local_json} if cfg.dataset == "squad" and self.squad_local_json else {}
            self._pools[pool_key] = D.load_pool(cfg.dataset, seed=cfg.seed, token_len_fn=len_fn,
                                                cache_dir=self.cache_dir, **kw)
            self._log_time("load_data", cfg.dataset, time.time() - t0, pool=len(self._pools[pool_key]))
        human = self._pools[pool_key][:cfg.n_samples]
        t0 = time.time()
        sampled = lm.sample(human, prompt_tokens=cfg.prompt_tokens, max_length=cfg.max_length,
                            min_length=cfg.min_length, min_words=cfg.min_words, temperature=cfg.temperature,
                            top_k=cfg.top_k, top_p=cfg.top_p, batch_size=cfg.gen_batch, seed=42 + cfg.seed)
        self._log_time("generate", cfg.data_key(), time.time() - t0, n=len(human))
        orig, samp = [], []
        for o, s in zip(human, sampled):
            o, s = D.trim_to_shorter_length(o, s)
            orig.append(o)
            samp.append(s)
        out = {"original": orig, "sampled": samp,
               "meta": {"source": D.LOADED_FROM.get(cfg.dataset, "?"), "cfg": asdict(cfg), "env": self.env}}
        with open(path, "w") as f:
            json.dump(out, f, ensure_ascii=False)
        return out

    def get_stats(self, cfg: RunCfg, pairs: dict) -> dict:
        path = os.path.join(self.out, "stats", cfg.stats_key() + ".json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        lm = self.lm(cfg.scorer_name, cfg.scorer_dtype or cfg.dtype)
        o, s = pairs["original"], pairs["sampled"]
        if cfg.truncate_words:
            o = [D.truncate_words(x, cfg.truncate_words) for x in o]
            s = [D.truncate_words(x, cfg.truncate_words) for x in s]
        t0 = time.time()
        st = {"original": lm.token_stats(o), "sampled": lm.token_stats(s)}
        self._log_time("score", cfg.stats_key(), time.time() - t0, n=len(o) * 2)
        with open(path, "w") as f:
            json.dump(st, f)
        return st

    def _score_texts_safe(self, lm, texts: list[str]) -> list[dict]:
        idx = [i for i, t in enumerate(texts) if t.strip()]
        res = lm.token_stats([texts[i] for i in idx]) if idx else []
        out = [{"ll": float("nan"), "logrank": float("nan")} for _ in texts]
        for i, r in zip(idx, res):
            out[i] = r
        return out

    def get_perturbations(self, cfg: RunCfg, pairs: dict) -> dict:
        k = max(cfg.n_pert_list)
        n = min(cfg.n_pert_texts, len(pairs["original"]))
        tkey = f"{cfg.data_key()}__{cfg.mask_model}__k{k}__n{n}__pct{cfg.pct_words_masked}__span{cfg.span_length}"
        path = os.path.join(self.out, "pert", tkey.replace("/", "_") + ".json")
        if os.path.exists(path):
            with open(path) as f:
                pert = json.load(f)
        else:
            filler = self.filler(cfg.mask_model)
            pert = {}
            t0 = time.time()
            for side in ["original", "sampled"]:
                rep = [x for x in pairs[side][:n] for _ in range(k)]
                pert[side] = perturb_texts(rep, filler.fill, span_length=cfg.span_length, chunk_size=40,
                                           pct=cfg.pct_words_masked, seed=cfg.seed + (0 if side == "original" else 1))
            self._log_time("perturb", tkey, time.time() - t0, n_texts=n, k=k)
            with open(path, "w") as f:
                json.dump(pert, f, ensure_ascii=False)
        # statistics of perturbed texts under the scorer (cached separately per scorer)
        spath = path.replace(".json", f"__scorer={cfg.scorer_name.replace('/', '_')}.json")
        if os.path.exists(spath):
            with open(spath) as f:
                return json.load(f)
        lm = self.lm(cfg.scorer_name, cfg.scorer_dtype or cfg.dtype)
        t0 = time.time()
        pst = {}
        for side in ["original", "sampled"]:
            st = self._score_texts_safe(lm, pert[side])
            pst[side] = {"ll": [[st[i * k + j]["ll"] for j in range(k)] for i in range(n)],
                         "logrank": [[st[i * k + j]["logrank"] for j in range(k)] for i in range(n)]}
        self._log_time("score_perturbed", tkey, time.time() - t0, n=2 * n * k)
        pst["n"], pst["k"] = n, k
        pst["examples"] = {side: pert[side][:3] for side in pert}
        with open(spath, "w") as f:
            json.dump(pst, f, ensure_ascii=False)
        return pst

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, cfg: RunCfg) -> list[dict]:
        rid = cfg.run_id()
        res_path = os.path.join(self.out, "results.jsonl")
        if os.path.exists(res_path):
            with open(res_path) as f:
                done = [json.loads(l) for l in f if l.strip()]
            prev = [r for r in done if r["run_id"] == rid]
            if prev:
                print(f"[skip] {rid} (already in results.jsonl)")
                return prev
        pairs = self.get_pairs(cfg)
        st = self.get_stats(cfg, pairs)
        rows, score_cols = [], {}
        base = {"run_id": rid, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                "scorer_name": cfg.scorer_name, "gpu": self.env.get("gpu")}
        H, Mc = st["original"], st["sampled"]
        for name, fn in M.FREE_FNS.items():
            h = [fn(x) for x in H]
            m = [fn(x) for x in Mc]
            score_cols[name] = (h, m)
            rows.append({**base, "method": name, "n_pert": 0, **MET.all_metrics(h, m, n_boot=self.n_boot)})
        if cfg.perturb:
            pst = self.get_perturbations(cfg, pairs)
            n = pst["n"]
            # perturbation-free methods on exactly the same subset, for a fair comparison
            for name in ["likelihood", "logrank", "LRR"]:
                h, m = score_cols[name][0][:n], score_cols[name][1][:n]
                rows.append({**base, "method": name + "@pert_subset", "n_pert": 0,
                             **MET.all_metrics(h, m, n_boot=self.n_boot)})
            for kk in cfg.n_pert_list:
                h = [M.detectgpt(H[i], pst["original"]["ll"][i], kk) for i in range(n)]
                m = [M.detectgpt(Mc[i], pst["sampled"]["ll"][i], kk) for i in range(n)]
                rows.append({**base, "method": "DetectGPT", "n_pert": kk, **MET.all_metrics(h, m, n_boot=self.n_boot)})
                score_cols[f"DetectGPT_{kk}"] = (h + [None] * (len(H) - n), m + [None] * (len(Mc) - n))
                h = [M.npr(H[i], pst["original"]["logrank"][i], kk) for i in range(n)]
                m = [M.npr(Mc[i], pst["sampled"]["logrank"][i], kk) for i in range(n)]
                rows.append({**base, "method": "NPR", "n_pert": kk, **MET.all_metrics(h, m, n_boot=self.n_boot)})
                score_cols[f"NPR_{kk}"] = (h + [None] * (len(H) - n), m + [None] * (len(Mc) - n))
        # per-text scores for error analysis
        with open(os.path.join(self.out, "scores", rid + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            cols = list(score_cols)
            w.writerow(["idx", "label", "n_tokens", "ll", "logrank", *cols, "text"])
            for lab, side, stats in [(0, "original", H), (1, "sampled", Mc)]:
                k = 0 if lab == 0 else 1
                for i, s in enumerate(stats):
                    w.writerow([i, lab, s.get("n_tokens"), s["ll"], s["logrank"],
                                *[score_cols[c][k][i] for c in cols], pairs[side][i]])
        with open(res_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        best = {r["method"] + (f"_{r['n_pert']}" if r["n_pert"] else ""): round(r["auroc"], 4) for r in rows}
        print(f"[done] {rid}: {best}")
        return rows

    def run_all(self, cfgs: list[RunCfg]):
        # group by generator/scorer to minimise model reloads. Cheapest/most important models first;
        # GPT-Neo-2.7B (10 GB of fp32 weights to load into Colab's 12 GB RAM) runs last, so a
        # memory crash there cannot cost any other result (finished runs are cached anyway).
        prio = ["gpt2-xl", "gpt2", "gpt2-onnx", "facebook/opt-2.7b", "Qwen/Qwen2.5-1.5B",
                "aubmindlab/aragpt2-medium", "EleutherAI/gpt-neo-2.7B"]
        rank = lambda m: prio.index(m) if m in prio else len(prio)
        cfgs = sorted(cfgs, key=lambda c: (rank(c.generator), rank(c.scorer_name), c.scorer_dtype or "", c.exp))
        out = []
        for i, c in enumerate(cfgs, 1):
            print(f"\n=== [{i}/{len(cfgs)}] {c.exp}: {c.dataset} | gen={c.generator} | scorer={c.scorer_name}")
            try:
                out.extend(self.evaluate(c))
            except Exception as e:  # noqa: BLE001 - one failing run must not kill a 2-hour session
                import traceback
                traceback.print_exc()
                with open(os.path.join(self.out, "errors.log"), "a") as f:
                    f.write(f"{c.run_id()}: {type(e).__name__}: {e}\n")
        return out
