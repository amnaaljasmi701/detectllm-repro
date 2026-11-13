#!/usr/bin/env python
"""Run the reproduction experiments.

Examples
--------
python scripts/run.py --preset smoke                # 3-5 min sanity check (gpt2 small, SQuAD)
python scripts/run.py --preset full                 # everything reported in the report (~2.5-3 h, T4)
python scripts/run.py --preset full --only E1 E2    # a subset of experiments
python scripts/run.py --preset full --list          # print the run list and exit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from repro.experiments import build, shard_of  # noqa: E402
from repro.pipeline import Runner     # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=["smoke", "quick", "full", "ci"])
    ap.add_argument("--shard", default=None, help="run only the configs of one CI job (see experiments.shard_of)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: all cores)")
    ap.add_argument("--only", nargs="*", default=None, help="experiment prefixes, e.g. E1 E2 E5")
    ap.add_argument("--out", default=None, help="output directory (default results/<preset>)")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cfgs = build(a.preset, a.only, a.shard)
    if a.list:
        for c in cfgs:
            print(f"[{shard_of(c)}]", c.exp, c.dataset, c.generator, "->", c.scorer_name, c.tag, f"seed={c.seed}")
        print(len(cfgs), "runs")
        sys.exit(0)
    try:
        import torch
        if not torch.cuda.is_available():
            torch.set_num_threads(a.threads or os.cpu_count())
            print(f"CPU run with {torch.get_num_threads()} threads")
    except ImportError:
        pass
    out = a.out or os.path.join("results", a.preset)
    Runner(out_dir=out, cache_dir=a.cache_dir, n_boot=a.n_boot).run_all(cfgs)
    print(f"\nAll done. Now run:  python scripts/aggregate.py --results {out}")
