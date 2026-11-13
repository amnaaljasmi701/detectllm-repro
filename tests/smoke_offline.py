"""Offline smoke test of the complete pipeline on CPU (no GPU, no HuggingFace).

GPT-2 small (124M, ONNX export from github.com/onnx/models) generates continuations of SQuAD
v1.1 dev paragraphs, scores them, and the perturbation-free detectors are evaluated.
This only checks that every stage works end to end; it is *not* one of the reproduction runs.

usage: python tests/smoke_offline.py ONNX ENCODER_JSON VOCAB_BPE SQUAD_DEV_JSON OUT_DIR N
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from offline_backend import OnnxGPT2          # noqa: E402
from repro.pipeline import Runner, RunCfg      # noqa: E402

onnx, enc, bpe, squad, out, n = sys.argv[1:7]
lm = OnnxGPT2(onnx, enc, bpe)
runner = Runner(out_dir=out, lm_factory=lambda name, dtype: lm,
                token_len_fn_factory=lambda ds, model: model.n_tokens, squad_local_json=squad, n_boot=500)
cfgs = [RunCfg("SMOKE", "squad", "gpt2-onnx", n_samples=int(n), gen_batch=int(n)),
        RunCfg("SMOKE-length", "squad", "gpt2-onnx", n_samples=int(n), gen_batch=int(n), truncate_words=50, tag="50w")]
runner.run_all(cfgs)
