"""The alpha-sweep core, for both trained adapters and closed-form maps.
"""

import gc
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from . import naming, storage
from .activations import (collect_activations, install_monkeypatch,
                          load_adapter, load_model)
from .probes import load_probe, score_batch, score_one

__all__ = [
    "ALPHA_SWEEP", "METADATA_COLS", "DATASET_ID", "PairSpec",
    "native_ceilings", "alpha_sweep", "run_pair_full",
    "fit_procrustes", "alpha_sweep_with_adapter",
    "load_test_set", "run_all", "replot_all",
]

ALPHA_SWEEP = [round(x * 0.1, 1) for x in range(11)]
METADATA_COLS = ("statement_type", "polarity", "topic")
DATASET_ID = "KingTechnician/tiu-splits"


@dataclass
class PairSpec:
    adapter_repo: str
    adapter_subfolder: str = "model"
    variant: str = "truth"

    def resolve(self):
        src_id, src_l, tgt_id, tgt_l = naming.parse_adapter_repo(self.adapter_repo)
        pid = naming.pair_id(src_id, src_l, tgt_id, tgt_l, self.variant)
        return src_id, src_l, tgt_id, tgt_l, pid


def load_test_set(statement_type=None):
    """Full test set as records, optionally filtered to one statement type.

    The published Table 7 runs filtered to statement_type == 'simple'; the
    rebuttal runs did not. Pass statement_type='simple' to reproduce the
    former. Returns (records, per_item_metadata, counts).
    """
    df = load_dataset(DATASET_ID)["test"].to_pandas()
    if statement_type is not None:
        df = df[df["statement_type"] == statement_type]
    keep = ["text", "label"] + [c for c in METADATA_COLS if c in df.columns]
    records = df[keep].to_dict("records")

    items = [
        {"label": int(r["label"]),
         **{c: (str(r[c]) if r.get(c) is not None else None)
            for c in METADATA_COLS if c in keep}}
        for r in records
    ]
    counts = {}
    for it in items:
        key = f"{it.get('statement_type', '?')}/{it.get('polarity', '?')}"
        counts[key] = counts.get(key, 0) + 1
    return records, items, counts


def native_ceilings(cached_src, cached_tgt, labels,
                    src_probe, src_dirs, tgt_probe, tgt_dirs):
    """Source and target probe performance on their own native activations.

    Returns (summary, per_item). The per_item half is what makes per-statement-
    type ceilings computable after the fact.
    """
    sc, ss, tc, ts = 0, [], 0, []
    sp_list, tp_list = [], []
    for s, t, lab in zip(cached_src, cached_tgt, labels):
        sp, sscore = score_one(src_probe, src_dirs, s)
        tp, tscore = score_one(tgt_probe, tgt_dirs, t)
        if sp == lab:
            sc += 1
        if tp == lab:
            tc += 1
        ss.append(sscore)
        ts.append(tscore)
        sp_list.append(int(sp))
        tp_list.append(int(tp))

    n = len(labels)
    summary = {
        "source": {"accuracy": sc / n, "auroc": float(roc_auc_score(labels, ss))},
        "target": {"accuracy": tc / n, "auroc": float(roc_auc_score(labels, ts))},
    }
    per_item = {
        "source_scores": ss, "target_scores": ts,
        "source_preds": sp_list, "target_preds": tp_list,
    }
    return summary, per_item


def _sweep_row(alpha, labels, all_s, preds, true_s, false_s, n):
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    pos = sum(int(p == 1) for p in preds)
    return {
        "alpha": alpha,
        "accuracy": correct / n,
        "auroc": float(roc_auc_score(labels, all_s)),
        "delta_mean": float(np.mean(true_s) - np.mean(false_s)),
        "pct_pred_positive": pos / n,
        "true_scores": true_s,
        "false_scores": false_s,
        "scores": all_s,   # per-item, dataset order — aligns with payload["items"]
        "preds": preds,    # per-item, dataset order
    }


def alpha_sweep(cached_tgt, cached_map, labels, tgt_probe, tgt_dirs,
                alphas=ALPHA_SWEEP):
    """Fuse target and mapped activations across alpha and score at each step."""
    rows, n = [], len(labels)
    for alpha in alphas:
        true_s, false_s, all_s, preds = [], [], [], []
        for t, m, lab in zip(cached_tgt, cached_map, labels):
            fused = (1 - alpha) * t + alpha * m
            pred, score = score_one(tgt_probe, tgt_dirs, fused)
            all_s.append(score)
            preds.append(int(pred))
            (true_s if lab == 1 else false_s).append(score)
        rows.append(_sweep_row(alpha, labels, all_s, preds, true_s, false_s, n))
    return rows


def run_pair_full(spec, src_id, src_l, tgt_id, tgt_l,
                  src_probe_repo, tgt_probe_repo, output_root,
                  statement_type=None, strict_probes=True):
    install_monkeypatch()
    src_tok, src_wrap = load_model(src_id)
    tgt_tok, tgt_wrap = load_model(tgt_id)
    mapper, adapter_cfg = load_adapter(spec.adapter_repo, spec.adapter_subfolder)
    tgt_probe, tgt_dirs = load_probe(tgt_probe_repo, strict=strict_probes)
    src_probe, src_dirs = load_probe(src_probe_repo, strict=strict_probes)

    records, items, counts = load_test_set(statement_type)
    print(f"test set: {len(items)} items -> {counts}")

    c_src, c_tgt, c_map, labels = collect_activations(
        src_wrap, tgt_wrap, src_tok, tgt_tok,
        f"model.layers.{src_l}", f"model.layers.{tgt_l}", mapper, records,
    )
    labels = [int(l) for l in labels]

    ceilings, native_items = native_ceilings(
        c_src, c_tgt, labels, src_probe, src_dirs, tgt_probe, tgt_dirs)
    sweep = alpha_sweep(c_tgt, c_map, labels, tgt_probe, tgt_dirs)

    pid = naming.pair_id(src_id, src_l, tgt_id, tgt_l, spec.variant)
    payload = {
        "pair_id": pid,
        "display_label": naming.display_pair_label(src_id, src_l, tgt_id, tgt_l),
        "source_model": src_id, "target_model": tgt_id,
        "source_layer": src_l, "target_layer": tgt_l,
        "variant": spec.variant,
        "adapter": {"repo": spec.adapter_repo,
                    "subfolder": spec.adapter_subfolder,
                    "config": adapter_cfg},
        "probes": {"source": src_probe_repo, "target": tgt_probe_repo},
        "n_test": len(labels),
        "statement_type_filter": statement_type,
        "ceiling": ceilings,
        "alpha_sweep": sweep,
        "items": items,
        "native_item_scores": native_items,
        "statement_counts": counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    storage.save_payload(output_root, pid, payload)

    del src_wrap, tgt_wrap, mapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def run_all(specs, output_root, statement_type=None, force=False, plot=True):
    """Sweep each PairSpec into output_root/<pair_id>/. Nothing is uploaded.

    Skips a pair whose sweep_results.json already exists unless force=True, so
    an interrupted session resumes where it stopped. Returns {pair_id: payload}.
    """
    import json
    from pathlib import Path
    from . import plots

    out = {}
    for spec in specs:
        src_id, src_l, tgt_id, tgt_l, pid = spec.resolve()
        path = storage.pair_dir(output_root, pid) / "sweep_results.json"
        if path.exists() and not force:
            print(f"skip {pid} (exists)")
            out[pid] = json.load(open(path))
            continue
        print(f"run  {pid}")
        payload = run_pair_full(
            spec, src_id, src_l, tgt_id, tgt_l,
            naming.probe_repo_for(src_id, src_l), naming.probe_repo_for(tgt_id, tgt_l),
            output_root, statement_type=statement_type,
        )
        if plot:
            plots.plot_all(payload, storage.pair_dir(output_root, pid))
        out[pid] = payload
    return out


def replot_all(output_root, pair_ids):
    """Re-render the three per-pair figures from saved JSON (no GPU)."""
    import json
    from . import plots

    for pid in pair_ids:
        path = storage.pair_dir(output_root, pid) / "sweep_results.json"
        if not path.exists():
            print(f"miss {pid}")
            continue
        plots.plot_all(json.load(open(path)), storage.pair_dir(output_root, pid))
        print(f"plot {pid}")


# --- closed-form baseline ---------------------------------------------------

def fit_procrustes(S, T):
    """Orthogonal Procrustes with centering and isotropic scale.

    f(s) = scale * (s - mu_s) @ Us @ R + mu_t. No learning, no hyperparameters.
    """
    mu_s = S.mean(axis=0, keepdims=True)
    mu_t = T.mean(axis=0, keepdims=True)
    Sc, Tc = S - mu_s, T - mu_t

    _, _, Vt_s = np.linalg.svd(Sc, full_matrices=False)
    Us = Vt_s.T
    Sr = Sc @ Us

    U, sv, Vt = np.linalg.svd(Sr.T @ Tc, full_matrices=False)
    R = U @ Vt
    scale = sv.sum() / (Sr ** 2).sum()

    def adapter(src):
        src = np.atleast_2d(src).astype(np.float32)
        return (scale * ((src - mu_s) @ Us @ R) + mu_t).astype(np.float32)

    adapter.scale = float(scale)
    adapter.k = int(Sr.shape[1])
    return adapter


def alpha_sweep_with_adapter(adapter_fn, src_eval, tgt_eval, probe, directions,
                             labels, alphas=ALPHA_SWEEP):
    """Vectorized alpha sweep for a callable source->target map.

    Emits the SAME row schema as `alpha_sweep`, unlike the original Procrustes
    notebook which stored only alpha/accuracy/auroc. That omission is why
    strict overshoot could not be computed for Procrustes runs; rows written
    from here support it.
    """
    mapped = adapter_fn(src_eval)
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    rows = []
    for alpha in alphas:
        fused = (1 - alpha) * tgt_eval + alpha * mapped
        preds, probs = score_batch(probe, directions, fused)
        all_s = [float(p) for p in probs]
        preds = [int(p) for p in preds]
        true_s = [s for s, l in zip(all_s, labels) if l == 1]
        false_s = [s for s, l in zip(all_s, labels) if l == 0]
        rows.append(_sweep_row(alpha, labels.tolist(), all_s, preds,
                               true_s, false_s, n))
    return rows