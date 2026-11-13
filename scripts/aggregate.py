#!/usr/bin/env python
"""Turn results/<preset>/results.jsonl into the tables, figures and error-analysis files used in the report.

python scripts/aggregate.py --results results/full
Outputs go to <results>/summary/ :
  E1_main.csv/.md, E1_claims.md, E1_tpr.md, E2_perturbation.csv/.md, E3_seeds.md,
  E4_decoding.md, E4_length.md, E4_precision.md, E5_arabic.md, timing.md,
  error_analysis.md, fig_*.png, summary.json
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repro.paper_numbers import PAPER, CLAIMS  # noqa: E402
from repro import metrics as MET               # noqa: E402

SHORT = {"gpt2-xl": "GPT2-xl", "EleutherAI/gpt-neo-2.7B": "Neo-2.7B", "facebook/opt-2.7b": "OPT-2.7B",
         "gpt2": "GPT2-small", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B", "aubmindlab/aragpt2-medium": "AraGPT2-M"}
DS = {"xsum": "XSum", "squad": "SQuAD", "writing": "WritingPrompts", "arwiki": "Arabic Wikipedia"}
FREE = ["likelihood", "rank", "logrank", "entropy", "LRR"]


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def pct(x):
    return None if x is None or pd.isna(x) else round(100 * x, 2)


def main(res_dir: str):
    out = os.path.join(res_dir, "summary")
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(res_dir, "results.jsonl")) if l.strip()]
    df = pd.DataFrame(rows)
    df["model"] = df["scorer_name"].map(lambda m: SHORT.get(m, m))
    df["gen"] = df["generator"].map(lambda m: SHORT.get(m, m))
    summary = {}

    # ---------------------------------------------------------------- E1 main table
    e1 = df[(df.exp == "E1") & (df.n_pert == 0)]
    recs = []
    for _, r in e1.iterrows():
        p = PAPER.get(r.dataset, {}).get(r.method, {}).get(r.generator)
        recs.append({"dataset": DS[r.dataset], "model": r.model, "method": r.method,
                     "ours": pct(r.auroc), "95% CI": f"[{pct(r.auroc_lo)}, {pct(r.auroc_hi)}]",
                     "paper": p, "diff": None if p is None else round(pct(r.auroc) - p, 2),
                     "TPR@1%FPR": pct(r["tpr@1%fpr"]), "TPR@5%FPR": pct(r["tpr@5%fpr"]), "n": r.n})
    t1 = pd.DataFrame(recs)
    if len(t1):
        t1.to_csv(os.path.join(out, "E1_main.csv"), index=False)
        piv = t1.pivot_table(index=["dataset", "method"], columns="model", values="ours").reset_index()
        piv_p = t1.pivot_table(index=["dataset", "method"], columns="model", values="paper").reset_index()
        with open(os.path.join(out, "E1_main.md"), "w") as f:
            f.write("## E1 - AUROC x100, ours\n\n" + md(piv.round(2)))
            f.write("\n## E1 - AUROC x100, paper (Table 1)\n\n" + md(piv_p.round(2)))
            f.write("\n## E1 - full rows\n\n" + md(t1))
        # claim: LRR vs best other perturbation-free method, per dataset x model
        cl = []
        for (d, m), g in t1.groupby(["dataset", "model"]):
            g = g.set_index("method")
            if "LRR" not in g.index:
                continue
            others = g.drop(index=[x for x in ["LRR"] if x in g.index])
            best_o = others["ours"].max()
            best_p = others["paper"].max() if others["paper"].notna().any() else None
            cl.append({"dataset": d, "model": m, "LRR ours": g.loc["LRR", "ours"], "best other ours": best_o,
                       "gain ours": round(g.loc["LRR", "ours"] - best_o, 2),
                       "LRR paper": g.loc["LRR", "paper"], "best other paper": best_p,
                       "gain paper": None if best_p is None or pd.isna(g.loc["LRR", "paper"]) else round(g.loc["LRR", "paper"] - best_p, 2),
                       "LRR is best (ours)": bool(g.loc["LRR", "ours"] >= best_o)})
        cl = pd.DataFrame(cl)
        with open(os.path.join(out, "E1_claims.md"), "w") as f:
            f.write("## Claim 1: LRR beats every other perturbation-free method\n\n" + md(cl))
            if len(cl):
                f.write(f"\nMean gain ours: {cl['gain ours'].mean():.2f} | mean gain paper (same 3 models): "
                        f"{cl['gain paper'].mean():.2f} | paper's 7-model claim: {CLAIMS['LRR_gain_over_best_perturbation_free']}\n")
                f.write(f"LRR best in {int(cl['LRR is best (ours)'].sum())}/{len(cl)} settings (ours)\n")
        summary["E1"] = t1.to_dict("records")
        summary["E1_claim"] = cl.to_dict("records")

        # figure: ours vs paper
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = t1.dropna(subset=["paper"])
    if len(t1) and len(sub):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        mk = {"likelihood": "o", "rank": "s", "logrank": "^", "entropy": "x", "LRR": "*"}
        col = {"XSum": "#1f77b4", "SQuAD": "#ff7f0e", "WritingPrompts": "#2ca02c"}
        for _, r in sub.iterrows():
            ax.scatter(r.paper, r.ours, marker=mk.get(r.method, "o"), color=col.get(r.dataset, "k"),
                       s=70 if r.method == "LRR" else 30)
        lim = [min(sub.paper.min(), sub.ours.min()) - 3, 101]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("AUROC reported in paper"); ax.set_ylabel("AUROC reproduced (ours)")
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker=v, ls="", color="gray", label=k) for k, v in mk.items()] + \
            [Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in col.items()]
        ax.legend(handles=h, fontsize=7, loc="lower right")
        ax.set_title("E1: reproduced vs reported AUROC")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_e1_scatter.png"), dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- E2 perturbation methods
    e2 = df[df.exp == "E2"]
    if len(e2):
        t2 = e2[["dataset", "method", "n_pert", "auroc", "auroc_lo", "auroc_hi", "tpr@1%fpr", "n"]].copy()
        for c in ["auroc", "auroc_lo", "auroc_hi", "tpr@1%fpr"]:
            t2[c] = t2[c].map(pct)
        t2["dataset"] = t2["dataset"].map(DS)
        t2 = t2[t2.method.isin(["DetectGPT", "NPR", "LRR@pert_subset", "logrank@pert_subset", "likelihood@pert_subset"])]
        t2.to_csv(os.path.join(out, "E2_perturbation.csv"), index=False)
        tests = []
        for d in e2.dataset.unique():
            kmax = e2[(e2.dataset == d) & (e2.method == "NPR")].n_pert.max()
            rid = e2[e2.dataset == d].run_id.iloc[0]
            sc = pd.read_csv(os.path.join(res_dir, "scores", rid + ".csv"))
            for a, b in [(f"NPR_{kmax}", f"DetectGPT_{kmax}"), (f"NPR_{kmax}", "LRR"), ("LRR", f"DetectGPT_{kmax}")]:
                s = sc.dropna(subset=[a, b])
                h1, m1 = s[s.label == 0][a].values, s[s.label == 1][a].values
                h2, m2 = s[s.label == 0][b].values, s[s.label == 1][b].values
                diff, lo, hi, p = MET.paired_bootstrap_diff(h1, m1, h2, m2)
                tests.append({"dataset": DS[d], "A": a, "B": b, "AUROC(A)-AUROC(B) x100": round(100 * diff, 2),
                              "95% CI": f"[{100*lo:.2f}, {100*hi:.2f}]", "P(A not better)": round(p, 3)})
        tests = pd.DataFrame(tests)
        with open(os.path.join(out, "E2_perturbation.md"), "w") as f:
            f.write("## E2 - perturbation-based methods (GPT2-xl scorer)\n\n" + md(t2))
            f.write("\n## Paired bootstrap tests (same texts)\n\n" + md(tests))
        summary["E2"] = t2.to_dict("records")
        summary["E2_tests"] = tests.to_dict("records")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(t2.dataset.unique()), figsize=(4 * len(t2.dataset.unique()), 3.3), squeeze=False)
        for ax, d in zip(axes[0], t2.dataset.unique()):
            for meth, c in [("DetectGPT", "#d62728"), ("NPR", "#1f77b4")]:
                s = t2[(t2.dataset == d) & (t2.method == meth)].sort_values("n_pert")
                ax.plot(s.n_pert, s.auroc, "o-", color=c, label=meth)
            lr = t2[(t2.dataset == d) & (t2.method == "LRR@pert_subset")]
            if len(lr):
                ax.axhline(lr.auroc.iloc[0], color="#2ca02c", ls="--", label="LRR (no perturbation)")
            ax.set_title(d); ax.set_xlabel("# perturbations"); ax.set_ylabel("AUROC")
            ax.set_xscale("log")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_e2_perturbations.png"), dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- E3 seeds
    ref = df[(df.exp == "E1") & (df.generator == df[df.exp == "E3"].generator.iloc[0])] if (df.exp == "E3").any() else None
    if ref is not None:
        e3 = pd.concat([ref[ref.dataset.isin(df[df.exp == "E3"].dataset.unique())], df[df.exp == "E3"]])
        e3 = e3[e3.method.isin(FREE)]
        g = e3.groupby(["dataset", "method"]).auroc.agg(["mean", "std", "min", "max", "count"]).reset_index()
        for c in ["mean", "std", "min", "max"]:
            g[c] = (100 * g[c]).round(2)
        g["dataset"] = g["dataset"].map(DS)
        rank_stab = []
        for (d, s), gg in e3.groupby(["dataset", "seed"]):
            rank_stab.append({"dataset": DS[d], "seed": s, "best method": gg.loc[gg.auroc.idxmax(), "method"]})
        with open(os.path.join(out, "E3_seeds.md"), "w") as f:
            f.write("## E3 - AUROC x100 across data/sampling seeds (seed 0 = E1)\n\n" + md(g))
            f.write("\n## Best perturbation-free method per seed\n\n" + md(pd.DataFrame(rank_stab)))
        summary["E3"] = g.to_dict("records")
        summary["E3_best"] = rank_stab

    # ---------------------------------------------------------------- E4
    for exp, fname in [("E4-decoding", "E4_decoding.md"), ("E4-length", "E4_length.md"), ("E4-precision", "E4_precision.md")]:
        e = df[(df.exp == exp) & df.method.isin(FREE)]
        if not len(e):
            continue
        base = df[(df.exp == "E1") & (df.generator == e.generator.iloc[0]) & df.dataset.isin(e.dataset.unique()) & df.method.isin(FREE)].copy()
        base["tag"] = "E1 default (T=1, top-k=50, full length, fp32)"
        e = pd.concat([base, e])
        t = e.pivot_table(index=["dataset", "tag"], columns="method", values="auroc").reset_index()
        for c in FREE:
            if c in t:
                t[c] = (100 * t[c]).round(2)
        t["dataset"] = t["dataset"].map(DS)
        t2 = e.pivot_table(index=["dataset", "tag"], columns="method", values="tpr@1%fpr").reset_index()
        for c in FREE:
            if c in t2:
                t2[c] = (100 * t2[c]).round(1)
        t2["dataset"] = t2["dataset"].map(DS)
        with open(os.path.join(out, fname), "w") as f:
            f.write(f"## {exp} - AUROC x100\n\n" + md(t) + f"\n## {exp} - TPR@1%FPR x100\n\n" + md(t2))
        summary[exp] = t.to_dict("records")
    e = df[(df.exp == "E4-length") & (df.method == "LRR")]
    if len(e):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for d in e.dataset.unique():
            s = e[e.dataset == d].sort_values("truncate_words")
            full = df[(df.exp == "E1") & (df.generator == s.generator.iloc[0]) & (df.dataset == d) & (df.method == "LRR")]
            xs = list(s.truncate_words) + ([150] if len(full) else [])
            ys = list(100 * s.auroc) + ([100 * full.auroc.iloc[0]] if len(full) else [])
            ax.plot(xs, ys, "o-", label=DS[d])
        ax.set_xlabel("words per text (150 = untruncated, approx.)"); ax.set_ylabel("LRR AUROC")
        ax.legend(fontsize=7); ax.set_title("E4: effect of text length")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_e4_length.png"), dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- E5 Arabic
    e5 = df[df.exp.str.startswith("E5") & df.method.isin(FREE)]
    if len(e5):
        t = e5.pivot_table(index=["exp", "dataset", "gen", "model"], columns="method", values="auroc").reset_index()
        for c in FREE:
            if c in t:
                t[c] = (100 * t[c]).round(2)
        t["dataset"] = t["dataset"].map(DS)
        t = t.rename(columns={"gen": "generator", "model": "scorer"})
        tt = e5.pivot_table(index=["exp", "dataset", "gen", "model"], columns="method", values="tpr@1%fpr").reset_index()
        for c in FREE:
            if c in tt:
                tt[c] = (100 * tt[c]).round(1)
        ci = e5[e5.method == "LRR"][["exp", "dataset", "gen", "model", "auroc", "auroc_lo", "auroc_hi"]].copy()
        for c in ["auroc", "auroc_lo", "auroc_hi"]:
            ci[c] = ci[c].map(pct)
        with open(os.path.join(out, "E5_arabic.md"), "w") as f:
            f.write("## E5 - AUROC x100\n\n" + md(t) + "\n## E5 - TPR@1%FPR x100\n\n" + md(tt) + "\n## LRR with 95% CI\n\n" + md(ci))
        summary["E5"] = t.to_dict("records")
        summary["E5_ci"] = ci.to_dict("records")
        import matplotlib.pyplot as plt
        sub = e5[e5.method.isin(["likelihood", "logrank", "LRR"])]
        sub = sub.assign(label=sub.apply(lambda r: f"{DS[r.dataset][:7]}\n{r.gen}->{r.model}", axis=1))
        piv = sub.pivot_table(index="label", columns="method", values="auroc") * 100
        ax = piv.plot.bar(figsize=(7, 3.4), rot=0, color=["#9ecae1", "#4292c6", "#08519c"])
        ax.set_ylim(40, 100); ax.set_ylabel("AUROC"); ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=6.5); ax.legend(fontsize=7)
        ax.set_title("E5: Arabic extension (generator -> scorer)")
        plt.tight_layout(); plt.savefig(os.path.join(out, "fig_e5_arabic.png"), dpi=200); plt.close()

    # ---------------------------------------------------------------- error analysis
    lines = ["# Error analysis (LRR)\n",
             "Highest-scoring human texts (false-positive candidates) and lowest-scoring machine texts "
             "(false-negative candidates). Threshold = the LRR value giving 5% FPR on that run.\n"]
    ea = []
    for exp, gen in [("E1", "gpt2-xl"), ("E1", "gpt2"), ("E5-arabic", "Qwen/Qwen2.5-1.5B"), ("E5-arabic", "aubmindlab/aragpt2-medium")]:
        for rid in df[(df.exp == exp) & (df.generator == gen)].run_id.unique():
            path = os.path.join(res_dir, "scores", rid + ".csv")
            if not os.path.exists(path):
                continue
            sc = pd.read_csv(path)
            d = df[df.run_id == rid].dataset.iloc[0]
            h = sc[sc.label == 0].sort_values("LRR", ascending=False)
            m = sc[sc.label == 1].sort_values("LRR")
            thr = np.quantile(sc[sc.label == 0].LRR.replace([np.inf], np.nan).dropna(), 0.95)
            fp_rate = float((sc[sc.label == 0].LRR > thr).mean())
            fn_rate = float((sc[sc.label == 1].LRR <= thr).mean())
            lines.append(f"\n## {DS[d]} | {SHORT.get(gen, gen)}  (threshold {thr:.3f}: FPR {fp_rate:.1%}, FNR {fn_rate:.1%})\n")
            # length/likelihood profile of errors vs correct
            mh = sc[sc.label == 1]
            fn = mh[mh.LRR <= thr]
            ok = mh[mh.LRR > thr]
            if len(fn) and len(ok):
                lines.append(f"Missed machine texts vs detected: mean tokens {fn.n_tokens.mean():.0f} vs {ok.n_tokens.mean():.0f}; "
                             f"mean log p {fn.ll.mean():.2f} vs {ok.ll.mean():.2f}; mean log-rank {fn.logrank.mean():.2f} vs {ok.logrank.mean():.2f}\n")
            for title, part in [("False-positive candidates (human, highest LRR)", h.head(3)),
                                ("False-negative candidates (machine, lowest LRR)", m.head(3))]:
                lines.append(f"\n**{title}**\n")
                for _, r in part.iterrows():
                    txt = str(r.text).replace("\n", " ")
                    lines.append(f"- LRR={r.LRR:.3f}, log p={r.ll:.2f}, log-rank={r.logrank:.2f}, tokens={r.n_tokens}: "
                                 f"\"{txt[:420]}{'...' if len(txt) > 420 else ''}\"\n")
                    ea.append({"dataset": d, "gen": gen, "type": title.split(" (")[0], "LRR": r.LRR, "ll": r.ll,
                               "logrank": r.logrank, "tokens": r.n_tokens, "text": txt[:600]})
    with open(os.path.join(out, "error_analysis.md"), "w") as f:
        f.writelines(lines)
    summary["error_analysis"] = ea

    # ---------------------------------------------------------------- timing / resources
    tp = os.path.join(res_dir, "timing.jsonl")
    if os.path.exists(tp):
        tm = pd.DataFrame([json.loads(l) for l in open(tp) if l.strip()])
        g = tm.groupby("stage").seconds.agg(["count", "sum", "mean"]).reset_index()
        g["sum_min"] = (g["sum"] / 60).round(1)
        g["mean"] = g["mean"].round(1)
        with open(os.path.join(out, "timing.md"), "w") as f:
            f.write(f"GPU: {tm.gpu.iloc[0]} | total {tm.seconds.sum()/3600:.2f} h\n\n" + md(g[["stage", "count", "mean", "sum_min"]]))
            gen = tm[tm.stage.isin(["generate", "score", "perturb", "score_perturbed"])][["stage", "what", "seconds"]]
            f.write("\n" + md(gen))
        summary["timing"] = {"gpu": tm.gpu.iloc[0], "total_hours": round(tm.seconds.sum() / 3600, 2),
                             "by_stage": g.to_dict("records")}
    if os.path.exists(os.path.join(res_dir, "errors.log")):
        summary["errors"] = open(os.path.join(res_dir, "errors.log")).read()
    # data sources actually used
    srcs = {}
    for p in glob.glob(os.path.join(res_dir, "data", "*.json")):
        js = json.load(open(p))
        srcs[os.path.basename(p)] = {"source": js["meta"].get("source"), "n": len(js["original"]),
                                     "mean_words_human": float(np.mean([len(x.split()) for x in js["original"]])),
                                     "example_machine": js["sampled"][0][:300]}
    summary["data"] = srcs
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
    print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/full")
    main(ap.parse_args().results)
