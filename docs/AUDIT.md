# Audit of the original implementation

Repository: <https://github.com/mbzuai-nlp/DetectLLM> (inspected November 2026, default branch).
Paper: Su, Zhuo, Wang & Nakov, *DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection
of Machine-Generated Text*, Findings of EMNLP 2023.

We read every file used by `main.py` and compared it with the paper. Findings are grouped by
their possible effect on the reported numbers.

## A. Differences between the paper and the code

| # | Finding | Where | Effect | What we do |
|---|---------|-------|--------|------------|
| A1 | The paper describes plain temperature sampling (T = 1). The code calls `model.generate(do_sample=True, temperature=...)` without `top_k`, and HuggingFace then applies its default `top_k = 50`. The machine texts are therefore top-50 samples, which are easier to detect than pure samples. | `sample_generate/generate.py` | Likely raises all AUROCs | Default `top_k=50` (matches the code); E4 adds true pure sampling (`top_k=0`), top-k 40, top-p 0.96 and T = 0.7 |
| A2 | The T5 mask-filling call likewise inherits `top_k = 50`. | `detectGPT.py::replace_masks` | Shapes the perturbation distribution | Kept (`top_k=50`), documented |
| A3 | The "machine" text starts with the first 30 tokens of the *human* text (the prompt), and both texts are scored in full, so 30 of roughly 150–200 scored tokens are identical human tokens. | `generate.py` | Dilutes the signal slightly. This is the standard DetectGPT protocol | Kept for faithfulness; discussed in the report |
| A4 | The paper reports AUROC only. | `all_baselines.py` | AUROC hides the low-FPR regime that matters in practice | We add TPR at 1% and 5% FPR, PR-AUC and bootstrap CIs |

## B. Implementation issues (bugs or fragile code)

| # | Finding | Where | Effect |
|---|---------|-------|--------|
| B1 | `optional_tok_kwargs['fast'] = False` is not a valid tokenizer argument (it should be `use_fast`), so OPT silently uses the fast tokenizer, contrary to the evident intent. | `utils/loadmodel.py` | Not measured here. The intent is not met. We keep the fast tokenizer that the code effectively used |
| B2 | `get_accurancy` returns the share of index-aligned pairs where the human score is lower than the machine score. That is not a classification accuracy. | `utils/run_baseline.py` | Mislabelled metric (not used in the paper tables) |
| B3 | LRR divides by the mean log-rank, which is 0 when every token is the model's arg-max. | `all_baselines.py` | Crash/inf on degenerate texts (rare at T = 1) |
| B4 | Batches are formed with `len(raw_data) // batch_size`, so the remainder is silently dropped when `n_samples` is not a multiple of `batch_size`. | `generate.py` | Fewer samples than requested in some configurations |
| B5 | A whole batch is re-sampled when any sample has fewer than 55 words, and the loop gives up after 4 tries, so short samples can survive. Human texts are then trimmed to the same short length. | `generate.py` | Some pairs are short; length affects detection (see E4) |
| B6 | `get_ll` places 13B/20B models on the *last* CUDA device by name matching. | `loss.py` | Portability only |
| B7 | Ranks come from a full `argsort` of the vocabulary, which is slow (O(V log V) per token) and breaks ties arbitrarily. | `rank.py` | Speed only. We count larger logits instead (identical without ties; `tests/test_stats.py`) |

## C. Reproducibility of the environment

* `requirements.txt` pins `transformers==4.28.0.dev0`, a development build that cannot be installed from PyPI, together with `datasets==2.10.0` and `huggingface-hub==0.11.1`. Current versions of `datasets` no longer run the script-based `xsum` loader the code relies on, so our data loaders try several equivalent sources and record which one was used.
* WritingPrompts must be downloaded by hand into `data/writingPrompts` (the README does not say so).
* The code fixes one seed (`torch.manual_seed(0/42)`) and reports one run of 300 pairs per cell. There is no variance estimate, so differences of 1–2 AUROC points between methods cannot be interpreted. E3 measures this variance.
* The hardware is stated (A100 40 GB), but the models are loaded in fp32 for ≤ 2.7B parameters. On a free 16 GB T4 we load the 2.7B models in fp16. E4 (precision) measures the effect of this change on GPT2-xl.
