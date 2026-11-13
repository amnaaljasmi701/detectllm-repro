# Reproducing DetectLLM: log-rank zero-shot detection of machine-generated text

This repository reproduces the central claims of

> Jinyan Su, Terry Yue Zhuo, Di Wang and Preslav Nakov. 2023. **DetectLLM: Leveraging Log Rank
> Information for Zero-Shot Detection of Machine-Generated Text.** *Findings of EMNLP 2023*, 12395–12412.
> [paper](https://aclanthology.org/2023.findings-emnlp.827) · [original code](https://github.com/mbzuai-nlp/DetectLLM)

It also extends them to **Arabic**. It was written for the AI504 *Reproduction and Validation*
assignment.

| Claim in the paper | Our experiment |
|---|---|
| C1. LRR (log-likelihood / log-rank ratio) beats every other perturbation-free zero-shot detector (+3.9 AUROC on average) | **E1**: 3 LLMs × 3 datasets × 5 detectors |
| C2. NPR (normalised log-rank perturbation) beats DetectGPT (+1.75 AUROC) and needs fewer perturbations | **E2**: GPT2-xl × 3 datasets, 1–20 T5 perturbations, paired bootstrap tests |
| (not tested in the paper) How stable are the numbers? | **E3**: data/sampling seeds; bootstrap CIs everywhere |
| (partly in the paper) What drives performance? | **E4**: decoding strategy, text length, fp16 vs fp32 |
| (new) Does LRR work for Arabic? | **E5**: Arabic Wikipedia with Qwen2.5-1.5B and AraGPT2, an English control and cross-model scoring |

## Quick start (Google Colab, free T4 GPU)

Open `notebooks/DetectLLM_Reproduction_Colab.ipynb` in Colab, choose *Runtime → Change runtime
type → T4 GPU* and run all cells. The notebook

1. installs the requirements (about 1 min),
2. runs the unit tests,
3. runs a 3–5 minute smoke test,
4. runs the full experiment suite (about 2.5–3 h on a T4), saving everything to Google Drive so a
   disconnect loses nothing (re-running a cell skips finished runs),
5. builds every table, figure and the error analysis, and zips them for download.

## Fully automatic run on GitHub Actions (no GPU needed)

`.github/workflows/reproduce.yml` runs the complete suite on free GitHub-hosted CPU runners
(4 vCPU, 16 GB RAM) for public repositories. It runs the unit tests, then 19 jobs (one per group
of runs that share generated data), and finally merges and aggregates the results and commits
them to the `results` branch. Start it from *Actions → Reproduce DetectLLM → Run workflow*, or push
a commit whose message contains `[run]`. On CPU the models run in fp32, as in the original paper.
CPU time limits the `ci` preset to 100 pairs per cell, 30 pairs × ≤ 10 perturbations for E2,
seed variance on XSum and three decoding variants (`python scripts/run.py --preset ci --list`).

## Command line

```bash
pip install -r requirements.txt            # + PyTorch with CUDA
bash run_all.sh smoke                      # sanity check, a few minutes
bash run_all.sh full results/full          # everything in the report
python scripts/run.py --preset full --only E5      # one experiment family
python scripts/run.py --preset full --list         # list the runs without executing
```

Presets: `smoke` (GPT-2 small, SQuAD, n = 20), `quick` (n = 100, ~1 h) and `full` (n = 300 as in
the paper, ~2.5–3 h on a T4).

## Repository layout

```
src/repro/
  stats.py         token log-prob / rank / entropy (numpy reference implementation)
  methods.py       likelihood, rank, log-rank, entropy, LRR, DetectGPT, NPR (higher = machine)
  metrics.py       AUROC, PR-AUC, TPR@FPR, bootstrap CIs, paired bootstrap tests
  data.py          XSum / SQuAD / WritingPrompts / Arabic Wikipedia, filtered as in the original
  models.py        batched HuggingFace scoring and sampling (GPU)
  perturb.py       T5 span-masking perturbations (ported from DetectGPT, seedable)
  pipeline.py      cached, resumable experiment runner (+ timing log for the cost analysis)
  experiments.py   E1–E5 and the presets
  paper_numbers.py Table 1 of the paper (for comparison)
scripts/run.py        run experiments
scripts/aggregate.py  tables (.md/.csv), figures, error analysis, summary.json
scripts/merge_shards.py  merges the per-job folders of the GitHub Actions run
tests/                unit tests (pytest); test_models_gpu.py checks the GPU code against
                      the original one-text-at-a-time implementation
tests/smoke_offline.py  CPU-only end-to-end check with an ONNX GPT-2 (no GPU or Hub access)
docs/AUDIT.md         line-by-line audit of the original code (paper/code differences, bugs)
```

## What is re-implemented and what is reused

The detectors, metrics, data filtering and sampling protocol were **re-implemented from the
paper and checked against the original code** (`tests/test_stats.py`, `tests/test_models_gpu.py`).
The perturbation helper functions (`tokenize_and_mask`, `extract_fills`, `apply_extracted_fills`)
are ports of the original DetectGPT/DetectLLM code (MIT licence), with a seeded RNG and a guard
for short texts. The WritingPrompts clean-up function is copied verbatim.

## Deviations from the original setup (and why)

| Original | Here | Reason |
|---|---|---|
| A100 40 GB, fp32 for ≤ 2.7B models | T4 16 GB, fp16 for the 2.7B models (GPT2-xl stays fp32) | free hardware; the effect is measured in E4 |
| T5-3B mask filler, 50–100 perturbations, 300 pairs | T5-large, ≤ 20 perturbations, 100 pairs (E2) | T5-3B + a 1.5B scorer exceed a T4's time budget; the paper also reports T5-large |
| 7 LLMs (GPT2-xl … NeoX-20B) | 3 LLMs (GPT2-xl, Neo-2.7B, OPT-2.7B) | GPU memory |
| one run, AUROC only | 3 seeds for GPT2-xl, bootstrap CIs, TPR@1%FPR | variance and low-FPR behaviour |
| `datasets==2.10` script loaders | parquet/hub equivalents (recorded in `results/*/data/*.json`) | old loaders no longer run |

## Outputs

`results/<preset>/summary/` contains `E1_main.md`, `E1_claims.md`, `E2_perturbation.md`,
`E3_seeds.md`, `E4_*.md`, `E5_arabic.md`, `error_analysis.md`, `timing.md`, the figures
`fig_*.png` and a machine-readable `summary.json`.

## Ethics

All data are public research corpora (XSum, SQuAD, WritingPrompts, Wikipedia). No personal data
is collected. Zero-shot detectors are **not** reliable enough to accuse individuals of using AI
(see the error analysis and TPR at 1% FPR), and false positives fall unevenly on some writers.
Do not use this code for high-stakes decisions about people.

## Citation

If you use this code, please cite the original paper (above) and this reproduction.
