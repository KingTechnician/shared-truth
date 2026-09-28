"""Model and adapter loading, the ModelWrapper hook patch, and activation
extraction.

Both extraction paths live here on purpose. They are not equivalent — see
`check_extraction_parity` — and the paper's central comparison runs across
them, so the difference should be visible rather than buried in two notebooks.
"""

import gc
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from representation_transfer.model_wrapper import ModelWrapper
from representation_transfer.autoencoder import AutoEncoder

__all__ = [
    "TOKENIZER_DEFAULTS", "install_monkeypatch", "load_model", "load_adapter",
    "collect_activations", "extract_last_token", "check_extraction_parity",
]

# The adapter sweep tokenized with no truncation and no max_length; the
# Procrustes path used truncation=True, max_length=512. Defaults below
# reproduce the adapter sweep, which is what the published numbers used.
TOKENIZER_DEFAULTS = {"truncation": False, "max_len": None}

# Procrustes-notebook settings, kept named so the difference is greppable.
PROCRUSTES_TOKENIZER = {"truncation": True, "max_len": 512}


def _patched_get_activation(self, layer_name):
    """Verbatim from the original sweep. Replaces ModelWrapper._get_activation
    so that replacement activations are spliced in by position rather than
    assumed to be the same length."""
    def hook(module, _input, output):
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        if layer_name in self.replacement_acts:
            copy_act = actual.clone()
            replacement = self.replacement_acts[layer_name]
            if copy_act.dim() == 2:
                copy_act = copy_act.unsqueeze(0)
            if replacement.dim() == 2:
                replacement = replacement.unsqueeze(0)
            if replacement.shape[1] > copy_act.shape[1]:
                copy_act = replacement[:, :copy_act.shape[1], :]
            else:
                copy_act[:, :replacement.shape[1], :] = replacement
            return (copy_act,) + output[1:] if is_tuple else copy_act
        self.activations[layer_name] = output
        return output
    return hook


def install_monkeypatch():
    ModelWrapper._get_activation = _patched_get_activation


def load_model(model_id, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device_map,
    )
    return tok, ModelWrapper(base, accelerator=None)


def load_adapter(repo_id, subfolder, device="cuda"):
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json", subfolder=subfolder)
    with open(cfg_path) as f:
        cfg = json.load(f)
    mapper = AutoEncoder(
        source_dim=cfg["source_dim"],
        target_dim=cfg["target_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device).half()
    weights = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", subfolder=subfolder)
    mapper.load_state_dict(torch.load(weights))
    mapper.eval()
    return mapper, cfg


def _encode(tok, text, truncation, max_len):
    kwargs = {"return_tensors": "pt"}
    if truncation:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_len
    return tok(text, **kwargs).to("cuda")


def _last_token(wrap, input_ids, layer_name):
    full = wrap.get_activations(input_ids, layer_name)
    wrap.remove_hooks()
    return full[0, -1, :].unsqueeze(0).half()


def collect_activations(src_wrap, tgt_wrap, src_tok, tgt_tok,
                        src_layer_name, tgt_layer_name, mapper, dataset,
                        truncation=TOKENIZER_DEFAULTS["truncation"],
                        max_len=TOKENIZER_DEFAULTS["max_len"]):
    """Paired source/target/mapped activations over a dataset of {text, label}.

    Both models are held in memory simultaneously and stepped together so the
    mapper can be applied inline. This is the path the published adapter sweeps
    used.
    """
    cached_src, cached_tgt, cached_map, labels = [], [], [], []
    for item in tqdm(dataset, desc="extracting"):
        s_in = _encode(src_tok, item["text"], truncation, max_len)
        t_in = _encode(tgt_tok, item["text"], truncation, max_len)
        with torch.no_grad():
            s_last = _last_token(src_wrap, s_in["input_ids"], src_layer_name)
            t_last = _last_token(tgt_wrap, t_in["input_ids"], tgt_layer_name)
            m_last = mapper(s_last)
        cached_src.append(s_last.cpu().numpy().astype(np.float32))
        cached_tgt.append(t_last.cpu().numpy().astype(np.float32))
        cached_map.append(m_last.cpu().numpy().astype(np.float32))
        labels.append(item["label"])
    return cached_src, cached_tgt, cached_map, labels


def extract_last_token(model_id, layer_idx, texts,
                       truncation=PROCRUSTES_TOKENIZER["truncation"],
                       max_len=PROCRUSTES_TOKENIZER["max_len"]):
    """One model at a time, loaded and freed within the call. Returns (n, d).

    Used by the Procrustes path, which needs full train-set matrices for the
    closed-form fit and cannot hold two models plus 2000 activations at once.
    Defaults reproduce the Procrustes notebook; pass **TOKENIZER_DEFAULTS to
    match the adapter sweep instead.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    wrap = ModelWrapper(base, accelerator=None)
    layer_name = f"model.layers.{layer_idx}"

    out = []
    for text in tqdm(texts, desc=f"{model_id.split('/')[-1]} L{layer_idx}"):
        ids = _encode(tok, text, truncation, max_len)
        with torch.no_grad():
            out.append(_last_token(wrap, ids["input_ids"], layer_name)
                       .cpu().numpy().astype(np.float32))

    del wrap, base, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(out)


def check_extraction_parity(model_id, layer_idx, texts, tol=0.0):
    """Do the two tokenizer settings yield identical last-token activations?

    The Procrustes notebook asserts in a comment that its extraction matches
    the adapter sweep's. It does not, strictly: the adapter path never
    truncates. For short TIU statements the two should agree exactly, but the
    paper's headline compares adapter runs against Procrustes runs, so this
    should be measured on the real eval texts rather than assumed.

    Returns a dict; max_abs_diff of 0.0 means true parity.
    """
    a = extract_last_token(model_id, layer_idx, texts, **TOKENIZER_DEFAULTS)
    b = extract_last_token(model_id, layer_idx, texts, **PROCRUSTES_TOKENIZER)
    diff = np.abs(a - b)
    n_rows = int((diff.max(axis=1) > tol).sum())
    result = {
        "model": model_id, "layer": layer_idx, "n_texts": len(texts),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "n_rows_differing": n_rows,
        "identical": bool(diff.max() <= tol),
    }
    print(f"parity {model_id} L{layer_idx}: max|Δ|={result['max_abs_diff']:.3e} "
          f"rows differing={n_rows}/{len(texts)}")
    return result
