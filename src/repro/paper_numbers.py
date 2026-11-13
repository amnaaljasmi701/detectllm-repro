"""AUROC (x100) reported in Table 1 of Su et al. (2023) for the three models we reproduce.

Transcribed from the ACL Anthology PDF (2023.findings-emnlp.827). DetectGPT and NPR use
T5-3B with 50 perturbations in the paper. Please double-check against the PDF if you edit.
"""
PAPER = {
    # dataset -> method -> {model: auroc}
    "xsum": {
        "likelihood": {"gpt2-xl": 89.16, "EleutherAI/gpt-neo-2.7B": 87.69, "facebook/opt-2.7b": 86.98},
        "rank":       {"gpt2-xl": 79.79, "EleutherAI/gpt-neo-2.7B": 77.87, "facebook/opt-2.7b": 76.07},
        "logrank":    {"gpt2-xl": 91.75, "EleutherAI/gpt-neo-2.7B": 90.79, "facebook/opt-2.7b": 89.18},
        "entropy":    {"gpt2-xl": 56.78, "EleutherAI/gpt-neo-2.7B": 55.14, "facebook/opt-2.7b": 50.34},
        "LRR":        {"gpt2-xl": 93.47, "EleutherAI/gpt-neo-2.7B": 92.24, "facebook/opt-2.7b": 88.70},
        "DetectGPT":  {"gpt2-xl": 98.80, "EleutherAI/gpt-neo-2.7B": 99.11, "facebook/opt-2.7b": 96.02},
        "NPR":        {"gpt2-xl": 99.40, "EleutherAI/gpt-neo-2.7B": 99.46, "facebook/opt-2.7b": 97.09},
    },
    "squad": {
        "likelihood": {"gpt2-xl": 90.72, "EleutherAI/gpt-neo-2.7B": 84.18, "facebook/opt-2.7b": 87.84},
        "rank":       {"gpt2-xl": 83.46, "EleutherAI/gpt-neo-2.7B": 79.77, "facebook/opt-2.7b": 81.85},
        "logrank":    {"gpt2-xl": 94.33, "EleutherAI/gpt-neo-2.7B": 89.52, "facebook/opt-2.7b": 91.76},
        "entropy":    {"gpt2-xl": 57.97, "EleutherAI/gpt-neo-2.7B": 58.48, "facebook/opt-2.7b": 53.29},
        "LRR":        {"gpt2-xl": 97.42, "EleutherAI/gpt-neo-2.7B": 95.74, "facebook/opt-2.7b": 95.89},
        "DetectGPT":  {"gpt2-xl": 98.52, "EleutherAI/gpt-neo-2.7B": 95.86, "facebook/opt-2.7b": 96.91},
        "NPR":        {"gpt2-xl": 99.40, "EleutherAI/gpt-neo-2.7B": 97.56, "facebook/opt-2.7b": 98.39},
    },
    "writing": {
        "likelihood": {"gpt2-xl": 96.71, "EleutherAI/gpt-neo-2.7B": 95.63, "facebook/opt-2.7b": 95.05},
        "rank":       {"gpt2-xl": 87.62, "EleutherAI/gpt-neo-2.7B": 82.79, "facebook/opt-2.7b": 83.89},
        "logrank":    {"gpt2-xl": 98.02, "EleutherAI/gpt-neo-2.7B": 97.15, "facebook/opt-2.7b": 96.32},
        "entropy":    {"gpt2-xl": 36.45, "EleutherAI/gpt-neo-2.7B": 34.07, "facebook/opt-2.7b": 39.75},
        "LRR":        {"gpt2-xl": 98.34, "EleutherAI/gpt-neo-2.7B": 98.02, "facebook/opt-2.7b": 96.45},
        "DetectGPT":  {"gpt2-xl": 99.30, "EleutherAI/gpt-neo-2.7B": 98.71, "facebook/opt-2.7b": 98.33},
        "NPR":        {"gpt2-xl": 99.78, "EleutherAI/gpt-neo-2.7B": 99.59, "facebook/opt-2.7b": 98.87},
    },
}
# 7-model averages (Table 1, "Avg." column) and the headline claims (abstract)
PAPER_AVG = {"xsum": {"LRR": 85.98, "NPR": 93.70, "DetectGPT": 92.80, "logrank": 83.83},
             "squad": {"LRR": 89.15, "NPR": 86.95, "DetectGPT": 84.92, "logrank": 80.88},
             "writing": {"LRR": 96.30, "NPR": 97.22, "DetectGPT": 94.90, "logrank": 95.02}}
CLAIMS = {"LRR_gain_over_best_perturbation_free": 3.9, "NPR_gain_over_DetectGPT": 1.75}
