#!/usr/bin/env python
"""Merge the per-job result folders written by the GitHub Actions workflow into one folder.

python scripts/merge_shards.py --shards results/shards --out results/full
Each sub-folder of --shards is one job (results.jsonl, timing.jsonl, data/, scores/, errors.log).
Rows are de-duplicated on (run_id, method, n_pert), because the E2 jobs start from a copy of the
GPT2-xl job's folder.
"""
import argparse
import glob
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="results/shards")
ap.add_argument("--out", default="results/full")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for sub in ["data", "scores"]:
    os.makedirs(os.path.join(a.out, sub), exist_ok=True)
seen, rows, timing, errors = set(), [], [], []
for d in sorted(glob.glob(os.path.join(a.shards, "*"))):
    shard = os.path.basename(d).replace("shard-", "")
    rp = os.path.join(d, "results.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["run_id"], r["method"], r["n_pert"])
            if key not in seen:
                seen.add(key)
                r["shard"] = shard
                rows.append(r)
    tp = os.path.join(d, "timing.jsonl")
    if os.path.exists(tp):
        for line in open(tp):
            if line.strip():
                t = json.loads(line)
                t["shard"] = shard
                timing.append(t)
    ep = os.path.join(d, "errors.log")
    if os.path.exists(ep):
        errors += [f"[{shard}] {l}" for l in open(ep) if l.strip()]
    for sub in ["data", "scores"]:
        for f in glob.glob(os.path.join(d, sub, "*")):
            shutil.copy(f, os.path.join(a.out, sub))
with open(os.path.join(a.out, "results.jsonl"), "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
with open(os.path.join(a.out, "timing.jsonl"), "w") as f:
    f.writelines(json.dumps(t) + "\n" for t in timing)
if errors:
    open(os.path.join(a.out, "errors.log"), "w").writelines(errors)
print(f"merged {len(rows)} rows from {len(glob.glob(os.path.join(a.shards, '*')))} jobs; {len(errors)} errors")
