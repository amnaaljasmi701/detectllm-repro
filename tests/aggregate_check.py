"""Exercise scripts/aggregate.py on a tiny offline run that has the same experiment names as the
real suite (E1, E2, E3, E4-*). A fake mask filler (random words) replaces T5. Numbers produced
here are meaningless; the point is that every table/figure code path runs.

usage: python tests/aggregate_check.py ONNX ENCODER_JSON VOCAB_BPE SQUAD_DEV_JSON OUT_DIR
"""
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from offline_backend import OnnxGPT2          # noqa: E402
from repro.pipeline import Runner, RunCfg      # noqa: E402

onnx, enc, bpe, squad, out = sys.argv[1:6]
lm = OnnxGPT2(onnx, enc, bpe)
WORDS = "the of and a to in is was for on that with as by at from his her".split()
rng = np.random.default_rng(0)


class FakeFiller:
    def fill(self, masked):
        res = []
        for t in masked:
            n = len(re.findall(r"<extra_id_\d+>", t))
            res.append("<pad> " + " ".join(f"<extra_id_{i}> {rng.choice(WORDS)} {rng.choice(WORDS)}" for i in range(n))
                       + f" <extra_id_{n}></s>")
        return res

    def free(self):
        pass


kw = dict(n_samples=6, gen_batch=6, max_length=70, min_length=60, min_words=20)
cfgs = [RunCfg("E1", "squad", "gpt2-onnx", **kw),
        RunCfg("E2", "squad", "gpt2-onnx", perturb=True, n_pert_texts=4, n_pert_list=(1, 3), mask_model="fake", **kw),
        RunCfg("E3", "squad", "gpt2-onnx", seed=1, **kw),
        RunCfg("E4-decoding", "squad", "gpt2-onnx", top_k=0, tag="pure", **kw),
        RunCfg("E4-length", "squad", "gpt2-onnx", truncate_words=25, tag="25w", **kw)]
Runner(out_dir=out, lm_factory=lambda n, d: lm, filler_factory=lambda n: FakeFiller(),
       token_len_fn_factory=lambda ds, m: m.n_tokens, squad_local_json=squad, n_boot=100).run_all(cfgs)
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "aggregate.py"),
                       "--results", out])
print(sorted(os.listdir(os.path.join(out, "summary"))))
