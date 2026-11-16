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
| C2. NPR (normalised log-rank perturbation) beats DetectGPT (+1.75 AUROC) and needs fewer perturbations | **E2**: GPT2-xl × 3 datasets, 1, 5 and 10 T5-large perturbations (30 pairs), paired bootstrap tests |
| (not tested in the paper) How stable are the numbers? | **E3**: data/sampling seeds; bootstrap CIs everywhere |
| (partly in the paper) What drives performance? | **E4**: decoding strategy (incl. truly untruncated sampling) and text length |
| (new) Does LRR work for Arabic? | **E5**: Arabic Wikipedia with Qwen2.5-1.5B and AraGPT2, an English control and cross-model scoring |

## Results at a glance (GitHub Actions run 43448901855, CPU, `ci` preset, 100 pairs per cell)

| | Finding |
|---|---|
| **C1 reproduced (direction)** | LRR is the best perturbation-free detector in 9/9 model × dataset settings and in 3/3 seeds. Our 45 AUROCs correlate r = 0.995 with Table 1 of the paper; 43/45 published values lie inside our 95% CIs. Mean gain over log-rank: 2.5 points (paper, same models: 1.9); significant in 5/9 settings. |
| **C2 reproduced (direction, 30 pairs)** | NPR > DetectGPT at 1, 5 and 10 perturbations on all three datasets (+3.2 to +11.8 at 10; significant on XSum only). LRR, with no perturbations, is statistically indistinguishable from NPR. |
| **Decoding matters most** | The original code always samples with top-k 50 (HuggingFace default), even in its top-p and temperature experiments. With the pure sampling the paper describes, LRR drops from 92.7 to **23.1** AUROC (below chance); with untruncated top-p 0.96, to 38.3. |
| **Length** | At 25 words every detector is at chance; 100 words give 89–95 AUROC. |
| **Low FPR** | At 1% FPR, LRR catches 38–81% of machine texts (best of all criteria everywhere). |
| **Arabic** | White-box LRR: 99.5 (AraGPT2) and 86.3 (Qwen2.5, vs 65.4 for log-likelihood). Cross-model: 63.8 and 23.9. |

Tables and figures: [`results/full/summary/`](results/full/summary/). Per-text scores and
generated texts: the [`results`](../../tree/results) branch.

## Fully automatic run on GitHub Actions (no GPU needed)

`.github/workflows/reproduce.yml` runs the complete suite on free GitHub-hosted CPU runners
(4 vCPU, 16 GB RAM) for public repositories. It runs the unit tests, then 19 jobs (one per group
of runs that share generated data), and finally merges and aggregates the results and commits
them to the `results` branch. Start it from *Actions → Reproduce DetectLLM → Run workflow*, or push
a commit whose message contains `[run]`. On CPU the models run in fp32, as in the original paper.
CPU time limits the `ci` preset to 100 pairs per cell, 30 pairs × ≤ 10 perturbations for E2,
seed variance on XSum and three decoding variants (`python scripts/run.py --preset ci --list`).

## Alternative: Google Colab (free T4 GPU)

`notebooks/DetectLLM_Reproduction_Colab.ipynb` runs the same code on a GPU: choose *Runtime →
Change runtime type → T4 GPU* and run all cells. It installs the requirements, runs the tests and a
smoke run, then the `full` preset (300 pairs, up to 20 perturbations, GPT2-xl in fp32 and the 2.7B
models in fp16; about 2.5–3 h), saving to Google Drive. **The numbers in the report come from the
GitHub Actions `ci` run, not from Colab.**

## Command line

```bash
pip install -r requirements.txt            # + PyTorch with CUDA
bash run_all.sh smoke                      # sanity check, a few minutes
bash run_all.sh full results/full          # everything in the report
python scripts/run.py --preset full --only E5      # one experiment family
python scripts/run.py --preset full --list         # list the runs without executing
```

Presets:

| preset | used for | pairs per cell | E2 | notes |
|---|---|---|---|---|
| `ci` | **the report** (GitHub Actions, CPU, fp32) | 100 | 30 pairs, 1/5/10 perturbations | E3 on XSum; 3 decoding variants |
| `full` | GPU users (Colab T4) | 300 | 100 pairs, 1/5/10/20 perturbations | adds fp16-vs-fp32 test |
| `quick` | GPU, ~1 h | 100 | 40 pairs, 1/5/10 | |
| `smoke` | sanity check, minutes | 20 | 6 pairs | GPT-2 small, SQuAD only |

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
tests/                unit tests (pytest). test_models_gpu.py checks the batched torch code against
                      the original one-text-at-a-time implementation; despite its name it runs on
                      CPU or GPU (it needs torch and internet access to download GPT-2, and is
                      skipped otherwise). All 21 tests passed on the GitHub Actions CPU runner.
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
| A100 40 GB GPUs, fp32 for ≤ 2.7B models | GitHub Actions CPU runners (4 vCPU, 16 GB), fp32 for all models | no GPU available; free, fully scripted |
| 300 pairs per cell | 100 pairs per cell | measured CPU cost (~2 h per 300 GPT2-xl pairs) and 5 concurrent jobs |
| T5-3B mask filler, 50 perturbations (main results) | T5-large, 1/5/10 perturbations, 30 pairs (E2) | perturbation is ~20x the cost of LRR; the paper itself reports T5-large results (its Table 3) |
| 7 LLMs (GPT2-xl … NeoX-20B) | 3 LLMs (GPT2-xl, Neo-2.7B, OPT-2.7B) + 2 Arabic | memory and time |
| one run, AUROC only | 3 seeds for GPT2-xl on XSum, bootstrap CIs, paired tests, TPR@1%FPR | variance and low-FPR behaviour |
| implicit top-k 50 in every decoding setting | top-k 50 by default + pure, top-p 0.96 (k = 0), T = 0.7 | to separate the effect of truncation |
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
