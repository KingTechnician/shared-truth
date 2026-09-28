"""Model registries, repo-name parsing, and canonical pair identifiers.


"""

import re

__all__ = [
    "MODEL_REGISTRY", "PROBE_SLUGS", "DISPLAY_NAMES", "ADAPTER_RE",
    "parse_adapter_repo", "probe_repo_for", "pair_id", "display_pair_label",
    "short_name", "adapter_tag", "parse_sweep_dir",
]

MODEL_REGISTRY = {
    "llama_3_8b_instruct":        "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama_3.1_8b_instruct":      "meta-llama/Llama-3.1-8B-Instruct",
    "llama_3.2_3b_instruct":      "meta-llama/Llama-3.2-3B-Instruct",
    "mistral_7b_instruct_v0.3":   "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen_2.5_1.5b_instruct":     "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_2.5_7b_instruct":       "Qwen/Qwen2.5-7B-Instruct",
    "gemma_7b_instruct":          "google/gemma-7b-it",
    "gemma_2_2b_instruct":        "google/gemma-2-2b-it",
}

PROBE_SLUGS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct":    "llama3.1-8b-instruct",
    "meta-llama/Llama-3.2-3B-Instruct":    "llama3.2-3b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3":  "mistral-7b-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct":          "qwen2.5-1.5b-instruct",
    "Qwen/Qwen2.5-7B-Instruct":            "qwen2.5-7b-instruct",
    "google/gemma-7b-it":                  "gemma-7b-instruct",
    "google/gemma-2-2b-it":                "gemma2-2b-instruct",
}

DISPLAY_NAMES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "meta-llama/Llama-3.1-8B-Instruct":    "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct":    "Llama-3.2-3B",
    "mistralai/Mistral-7B-Instruct-v0.3":  "Mistral-7B",
    "Qwen/Qwen2.5-1.5B-Instruct":          "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B-Instruct":            "Qwen2.5-7B",
    "google/gemma-7b-it":                  "Gemma-7B",
    "google/gemma-2-2b-it":                "Gemma-2-2B",
}

# The optional _seed_NN suffix was a Cell 4 monkeypatch of the Cell 3 regex in
# the adapter notebook. Folded in here so there is one pattern, not two.
ADAPTER_RE = re.compile(
    r"^(?P<src>[a-z0-9_.]+?)_l(?P<src_l>\d+)_to_(?P<tgt>[a-z0-9_.]+?)_l(?P<tgt_l>\d+)"
    r"(?:_seed_(?P<seed>\d+))?$"
)

# pair_id shape: <src_slug>_l<N>_to_<tgt_slug>_l<M>__<variant>, dots -> underscores
SWEEP_DIR_RE = re.compile(
    r"^(?P<src>[a-z0-9_\-]+?)_l(?P<src_l>\d+)_to_(?P<tgt>[a-z0-9_\-]+?)_l(?P<tgt_l>\d+)"
    r"__(?P<variant>.+)$"
)

_SLUG_TO_MODEL = {v.replace(".", "_"): k for k, v in PROBE_SLUGS.items()}


def parse_adapter_repo(repo_id):
    """'owner/gemma_2_2b_instruct_l13_to_llama_3.2_3b_instruct_l12_seed_42'
    -> (src_model_id, src_layer, tgt_model_id, tgt_layer).

    The seed suffix is matched and discarded — it identifies the adapter
    checkpoint, not the model pair.
    """
    name = repo_id.split("/", 1)[-1]
    m = ADAPTER_RE.match(name)
    if not m:
        raise ValueError(f"adapter repo does not match expected pattern: {name}")
    src_short, tgt_short = m["src"], m["tgt"]
    if src_short not in MODEL_REGISTRY or tgt_short not in MODEL_REGISTRY:
        raise KeyError(f"unknown short name in {name}: {src_short!r} / {tgt_short!r}")
    return (MODEL_REGISTRY[src_short], int(m["src_l"]),
            MODEL_REGISTRY[tgt_short], int(m["tgt_l"]))


def adapter_seed(repo_id):
    """Seed integer from an adapter repo name, or None for a base repo."""
    m = ADAPTER_RE.match(repo_id.split("/", 1)[-1])
    if not m:
        raise ValueError(f"adapter repo does not match expected pattern: {repo_id}")
    return int(m["seed"]) if m["seed"] else None


def probe_repo_for(model_id, layer, owner="KingTechnician"):
    return f"{owner}/tiu-acts-{PROBE_SLUGS[model_id]}-layer-{layer}-probe"


def pair_id(src_id, src_l, tgt_id, tgt_l, variant="truth"):
    s = PROBE_SLUGS[src_id].replace(".", "_")
    t = PROBE_SLUGS[tgt_id].replace(".", "_")
    return f"{s}_l{src_l}_to_{t}_l{tgt_l}__{variant}"


def display_pair_label(src_id, src_l, tgt_id, tgt_l):
    return f"{DISPLAY_NAMES[src_id]} (L{src_l}) \u2192 {DISPLAY_NAMES[tgt_id]} (L{tgt_l})"


def short_name(model_id):
    """Display name, falling back to the bare repo name for unregistered ids."""
    return DISPLAY_NAMES.get(model_id, model_id.split("/")[-1])


def adapter_tag(variant):
    """'stmt_seed42' -> 'seed42'.

    Observed values: origadapter | seed42 | seed43 | seed44 | randbase | shufbase.
    """
    return variant.replace("stmt_", "")


def parse_sweep_dir(dirname):
    """Inverse of pair_id: directory name -> dict with model ids, layers, variant.

    Returns None if the name does not parse, so callers can glob a directory
    and skip non-pair entries (activation_dumps/, analysis_*/) without a
    try/except around every iteration.
    """
    m = SWEEP_DIR_RE.match(dirname)
    if not m:
        return None
    src = _SLUG_TO_MODEL.get(m["src"])
    tgt = _SLUG_TO_MODEL.get(m["tgt"])
    if src is None or tgt is None:
        return None
    return {
        "src_id": src, "src_layer": int(m["src_l"]),
        "tgt_id": tgt, "tgt_layer": int(m["tgt_l"]),
        "variant": m["variant"], "tag": adapter_tag(m["variant"]),
        "pair_id": dirname,
    }
