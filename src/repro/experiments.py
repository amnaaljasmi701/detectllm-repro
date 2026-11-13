"""Experiment definitions.

E1  main reproduction       3 scoring/generating LLMs x 3 datasets, perturbation-free methods (Table 1)
E2  perturbation methods    GPT2-xl x 3 datasets: DetectGPT vs NPR with 1..20 T5 perturbations (Table 1/3)
E3  seed variance           GPT2-xl on XSum and SQuAD with two extra data/sampling seeds
E4  sensitivity             decoding strategy, text length, numerical precision (GPT2-xl, XSum)
E5  Arabic extension        LRR on Arabic Wikipedia with Arabic-capable LLMs, an English control,
                            and cross-model (black-box-like) scoring
Presets scale the sample sizes: ``smoke`` (minutes, gpt2 small), ``quick`` (~1 h on a T4),
``full`` (~2.5-3 h on a T4; the setting reported in the paper uses n=300).
"""
from __future__ import annotations

from .pipeline import RunCfg

PAPER_MODELS = ["gpt2-xl", "EleutherAI/gpt-neo-2.7B", "facebook/opt-2.7b"]
DATASETS = ["xsum", "squad", "writing"]
AR_MODELS = ["Qwen/Qwen2.5-1.5B", "aubmindlab/aragpt2-medium"]

PRESETS = {
    "smoke": dict(n=20, n_pert_texts=6, n_pert_list=(1, 2), mask="t5-small", models=["gpt2"],
                  datasets=["squad"], ar_models=[], seeds=[1], ar_n=20),
    "quick": dict(n=100, n_pert_texts=40, n_pert_list=(1, 5, 10), mask="t5-large", models=PAPER_MODELS,
                  datasets=DATASETS, ar_models=AR_MODELS, seeds=[1, 2], ar_n=100),
    "full": dict(n=300, n_pert_texts=100, n_pert_list=(1, 5, 10, 20), mask="t5-large", models=PAPER_MODELS,
                 datasets=DATASETS, ar_models=AR_MODELS, seeds=[1, 2], ar_n=300),
    # CPU-only GitHub Actions runners (4 vCPU, 16 GB RAM, 6 h/job): same as "full" except the
    # perturbation experiment, which is ~20x more expensive than the rest, and no fp16 test
    # (fp16 is not a meaningful CPU setting).
    "ci": dict(n=300, n_pert_texts=50, n_pert_list=(1, 5, 10), mask="t5-large", models=PAPER_MODELS,
               datasets=DATASETS, ar_models=AR_MODELS, seeds=[1, 2], ar_n=300, no_precision=True),
}

SHORT = {"gpt2-xl": "gptxl", "gpt2": "gpt2", "EleutherAI/gpt-neo-2.7B": "neo", "facebook/opt-2.7b": "opt"}


def shard_of(c: RunCfg) -> str:
    """Name of the CI job that runs this config (runs that share generated data share a job)."""
    if c.exp.startswith("E5"):
        return "ar-control" if c.exp == "E5-control" else "ar-main"
    if c.exp == "E2":
        return f"e2-{c.dataset}"
    if c.exp == "E3":
        return f"seed{c.seed}-{c.dataset}"
    if c.exp == "E4-decoding":
        return f"dec-{c.tag}"
    return f"{SHORT.get(c.generator, c.generator)}-{c.dataset}"   # E1, E4-length, E4-precision


def build(preset: str = "full", only: list[str] | None = None, shard: str | None = None) -> list[RunCfg]:
    P = PRESETS[preset]
    n = P["n"]
    ref = P["models"][0]               # gpt2-xl (or gpt2 in the smoke preset)
    cfgs: list[RunCfg] = []

    # E1 - main table, perturbation-free methods
    for m in P["models"]:
        for d in P["datasets"]:
            cfgs.append(RunCfg("E1", d, m, n_samples=n))

    # E2 - perturbation-based methods (re-uses the E1 GPT2-xl data and statistics)
    for d in P["datasets"]:
        cfgs.append(RunCfg("E2", d, ref, n_samples=n, perturb=True, n_pert_texts=P["n_pert_texts"],
                           n_pert_list=P["n_pert_list"], mask_model=P["mask"]))

    # E3 - seeds (seed 0 is E1)
    for s in P["seeds"]:
        for d in P["datasets"][:2]:
            cfgs.append(RunCfg("E3", d, ref, n_samples=n, seed=s))

    # E4 - sensitivity on XSum
    d0 = P["datasets"][0]
    for (T, k, p, tag) in [(1.0, 0, 1.0, "pure"), (1.0, 40, 1.0, "topk40"), (1.0, 0, 0.96, "topp0.96"),
                           (0.7, 50, 1.0, "T0.7")]:
        cfgs.append(RunCfg("E4-decoding", d0, ref, n_samples=n, temperature=T, top_k=k, top_p=p, tag=tag))
    for w in [25, 50, 100]:
        for d in P["datasets"]:
            cfgs.append(RunCfg("E4-length", d, ref, n_samples=n, truncate_words=w, tag=f"{w}w"))
    if preset != "smoke" and not P.get("no_precision"):
        cfgs.append(RunCfg("E4-precision", d0, ref, n_samples=n, scorer_dtype="fp16", tag="fp16-scoring"))

    # E5 - Arabic extension
    for m in P["ar_models"]:
        cfgs.append(RunCfg("E5-arabic", "arwiki", m, n_samples=P["ar_n"], gen_batch=10))
    if P["ar_models"]:
        q, a = P["ar_models"]
        cfgs.append(RunCfg("E5-control", d0, q, n_samples=P["ar_n"], gen_batch=10, tag="english-control"))
        cfgs.append(RunCfg("E5-cross", "arwiki", q, scorer=a, n_samples=P["ar_n"], gen_batch=10, tag="cross"))
        cfgs.append(RunCfg("E5-cross", "arwiki", a, scorer=q, n_samples=P["ar_n"], gen_batch=10, tag="cross"))

    if only:
        cfgs = [c for c in cfgs if any(c.exp.startswith(o) for o in only)]
    if shard:
        cfgs = [c for c in cfgs if shard_of(c) == shard]
    return cfgs
