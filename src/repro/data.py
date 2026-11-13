"""Human-written text pools, filtered exactly as in the original repository
(``baselines/sample_generate/generate.py::generate_data``):

1. remove duplicates (order-preserving), strip whitespace, collapse newlines;
2. keep only texts with > 250 words (if any exist);
3. shuffle with ``random.seed(seed)`` (seed 0 = original) and keep the first 5,000;
4. keep only texts with <= 512 tokens under the *preprocessing* tokenizer
   (t5-small in the original; for Arabic we use <= 512 *words*, because the English T5
   vocabulary cannot represent Arabic script and a model-independent filter keeps the same
   human texts for every Arabic model).

Dataset sources: the original code used HuggingFace ``datasets`` 2.10 with the canonical
``xsum`` / ``squad`` scripts and the fairseq WritingPrompts files. Hub layouts have changed
since 2023, so each loader tries several equivalent sources and records which one worked.
"""
from __future__ import annotations

import json
import os
import random
import tarfile
import urllib.request
from typing import Callable, Optional

LOADED_FROM: dict[str, str] = {}


# ----------------------------------------------------------------------------- text utils
def strip_newlines(text: str) -> str:
    return " ".join(text.split())


def trim_to_shorter_length(a: str, b: str) -> tuple[str, str]:
    """Truncate both texts to the word length of the shorter one (original behaviour)."""
    n = min(len(a.split(" ")), len(b.split(" ")))
    return " ".join(a.split(" ")[:n]), " ".join(b.split(" ")[:n])


def truncate_words(text: str, n_words: int) -> str:
    return " ".join(text.split(" ")[:n_words])


# WritingPrompts clean-up copied from the original custom_datasets.py
def process_prompt(prompt: str) -> str:
    return prompt.replace("[ WP ]", "").replace("[ OT ]", "")


def process_spaces(story: str) -> str:
    return story.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(
        " ;", ";").replace(" '", "'").replace(" ’ ", "'").replace(" :", ":").replace("<newline>", "\n").replace(
        "`` ", '"').replace(" ''", '"').replace("''", '"').replace(".. ", "... ").replace(" )", ")").replace(
        "( ", "(").replace(" n't", "n't").replace(" i ", " I ").replace(" i'", " I'").replace("\\'", "'").replace(
        "\n ", "\n").strip()


# ----------------------------------------------------------------------------- raw loaders
def _try_hf(attempts: list[tuple[tuple, dict]], field: str, tag: str) -> list[str]:
    import datasets
    errors = []
    for args, kw in attempts:
        try:
            ds = datasets.load_dataset(*args, **kw)
            LOADED_FROM[tag] = f"load_dataset{args} {kw}"
            return list(ds[field])
        except Exception as e:  # noqa: BLE001 - we want to try the next source
            errors.append(f"{args} {kw}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"Could not load {tag}. Tried:\n  " + "\n  ".join(errors) +
                       "\nTip: `pip install 'datasets<3'` restores script-based datasets.")


def load_xsum_raw(cache_dir=None) -> list[str]:
    return _try_hf([
        (("EdinburghNLP/xsum",), dict(split="train", cache_dir=cache_dir)),
        (("EdinburghNLP/xsum",), dict(split="train", cache_dir=cache_dir, revision="refs/convert/parquet")),
        (("EdinburghNLP/xsum",), dict(split="train", cache_dir=cache_dir, trust_remote_code=True)),
        (("xsum",), dict(split="train", cache_dir=cache_dir, trust_remote_code=True)),
    ], "document", "xsum")


def load_squad_raw(cache_dir=None, local_json: Optional[str] = None) -> list[str]:
    if local_json:  # SQuAD v1.1 json (used by the offline smoke test)
        with open(local_json) as f:
            js = json.load(f)
        LOADED_FROM["squad"] = f"local:{local_json}"
        return [p["context"] for art in js["data"] for p in art["paragraphs"]]
    return _try_hf([
        (("rajpurkar/squad",), dict(split="train", cache_dir=cache_dir)),
        (("squad",), dict(split="train", cache_dir=cache_dir)),
    ], "context", "squad")


WP_URL = "https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz"


def load_writing_raw(cache_dir=None) -> list[str]:
    """Validation split of WritingPrompts, prompt + story joined, NSFW removed (original code)."""
    cache_dir = cache_dir or os.path.expanduser("~/.cache/repro")
    root = os.path.join(cache_dir, "writingPrompts")
    src, tgt = os.path.join(root, "valid.wp_source"), os.path.join(root, "valid.wp_target")
    try:
        if not os.path.exists(tgt):
            os.makedirs(cache_dir, exist_ok=True)
            tar_path = os.path.join(cache_dir, "writingPrompts.tar.gz")
            if not os.path.exists(tar_path):
                urllib.request.urlretrieve(WP_URL, tar_path)
            with tarfile.open(tar_path) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(("valid.wp_source", "valid.wp_target")):
                        tf.extract(m, cache_dir)
        with open(src) as f:
            prompts = f.readlines()
        with open(tgt) as f:
            stories = f.readlines()
        LOADED_FROM["writing"] = "fairseq writingPrompts.tar.gz (valid split)"
    except Exception as e:  # noqa: BLE001
        print(f"[data] fairseq download failed ({e}); falling back to HF hub copy")
        import datasets
        ds = datasets.load_dataset("euclaise/writingprompts", split="validation", cache_dir=cache_dir)
        prompts, stories = list(ds["prompt"]), list(ds["story"])
        LOADED_FROM["writing"] = "load_dataset('euclaise/writingprompts', split='validation')"
    prompts = [process_prompt(p) for p in prompts]
    joined = [process_spaces(p + " " + s) for p, s in zip(prompts, stories)]
    filtered = [s for s in joined if "nsfw" not in s and "NSFW" not in s]
    random.seed(0)
    random.shuffle(filtered)
    return filtered


def load_arwiki_raw(cache_dir=None, max_articles: int = 30000) -> list[str]:
    """Arabic Wikipedia articles (extension experiment). Streams the first ``max_articles``."""
    import datasets
    errors = []
    for args in [("wikimedia/wikipedia", "20231101.ar"), ("wikipedia", "20220301.ar")]:
        try:
            ds = datasets.load_dataset(*args, split="train", streaming=True, cache_dir=cache_dir)
            out = []
            for i, ex in enumerate(ds):
                out.append(ex["text"])
                if i + 1 >= max_articles:
                    break
            LOADED_FROM["arwiki"] = f"load_dataset{args} streaming, first {len(out)} articles"
            return out
        except Exception as e:  # noqa: BLE001
            errors.append(f"{args}: {e}")
    raise RuntimeError("Could not load Arabic Wikipedia: " + "; ".join(errors))


RAW_LOADERS = {"xsum": load_xsum_raw, "squad": load_squad_raw, "writing": load_writing_raw,
               "arwiki": load_arwiki_raw}


# ----------------------------------------------------------------------------- filtering
def prepare_pool(texts: list[str], dataset: str, seed: int = 0,
                 token_len_fn: Optional[Callable[[list[str]], list[int]]] = None,
                 max_tokens: int = 512, pool_size: int = 5000, min_words: int = 250) -> list[str]:
    data = list(dict.fromkeys(texts))
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    if dataset in ("writing", "squad", "xsum", "arwiki"):
        long_data = [x for x in data if len(x.split()) > min_words]
        if long_data:
            data = long_data
    random.seed(seed)
    random.shuffle(data)
    data = data[:pool_size]
    if token_len_fn is not None:
        lens = token_len_fn(data)
        data = [x for x, n in zip(data, lens) if n <= max_tokens]
    return data


def load_pool(dataset: str, seed: int = 0, token_len_fn=None, cache_dir=None, **kw) -> list[str]:
    if dataset == "squad" and kw.get("local_json"):
        raw = load_squad_raw(cache_dir, local_json=kw["local_json"])
    else:
        raw = RAW_LOADERS[dataset](cache_dir)
    return prepare_pool(raw, dataset, seed=seed, token_len_fn=token_len_fn)
