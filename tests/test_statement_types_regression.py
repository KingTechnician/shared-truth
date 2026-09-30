"""Regression tests for shared_truth.residuals.statement_type_tables (Tables A-D).

1. Offline equivalence (no network): the module against a verbatim copy of the
   experiment notebook's Cell 10 on synthetic payloads, including Table D's
   RNG consumption order.

2. Published regression (needs the results repo): recomputes the tables from
   the 12 full-test-set trained runs and compares them with
   sweep-results/analysis_statement_types/*.csv.

       HF_TOKEN=... pytest tests/test_statement_types_regression.py -v

   Set SHARED_TRUTH_RESULTS_DIR to an existing snapshot to skip the download.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from shared_truth import naming, residuals, storage


# --- 1. offline equivalence ---------------------------------------------------

N_BOOT_OFFLINE = 40   # the offline checks pin logic and RNG order, not the resample count


def _original_cell10(payloads, N_BOOT=1000):
    """Verbatim logic of Cell 10, returning frames instead of writing CSVs.
    (N_BOOT is a parameter here only so the offline test runs in seconds.)"""
    PLATEAU_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    KEY_ALPHAS = [0.0, 0.7, 1.0]
    PLATEAU_TOL = 0.02
    COLLAPSE_AUROC = 0.60
    MIN_N = 20
    RNG = np.random.default_rng(0)

    def statement_group(item):
        if (item.get("polarity") or "").lower() == "negated":
            return "negated"
        return item.get("statement_type") or "unknown"

    def safe_auroc(labels, scores):
        labels = np.asarray(labels); scores = np.asarray(scores)
        if len(labels) < MIN_N or len(np.unique(labels)) < 2:
            return np.nan
        return float(roc_auc_score(labels, scores))

    def boot_auroc_ci(labels, scores, n_boot=N_BOOT):
        labels = np.asarray(labels); scores = np.asarray(scores)
        if len(labels) < MIN_N or len(np.unique(labels)) < 2:
            return (np.nan, np.nan)
        idx_pool = np.arange(len(labels))
        stats = []
        for _ in range(n_boot):
            idx = RNG.choice(idx_pool, size=len(idx_pool), replace=True)
            if len(np.unique(labels[idx])) < 2:
                continue
            stats.append(roc_auc_score(labels[idx], scores[idx]))
        if not stats:
            return (np.nan, np.nan)
        return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

    def adapter_tag(variant):
        return variant.replace("stmt_", "")

    long_rows, ci_rows, struct_rows = [], [], []
    for pl in payloads:
        traj = pl["display_label"]
        tag = adapter_tag(pl["variant"])
        items = pl["items"]
        labels = np.array([it["label"] for it in items])
        groups = np.array([statement_group(it) for it in items])
        group_names = ["all"] + sorted(set(groups))
        sweep = {round(r["alpha"], 1): r for r in pl["alpha_sweep"]}
        for g in group_names:
            mask = np.ones(len(items), bool) if g == "all" else (groups == g)
            n = int(mask.sum())
            if n < MIN_N:
                continue
            g_labels = labels[mask]
            per_alpha = {}
            for a, row in sweep.items():
                scores = np.array(row["scores"])[mask]
                preds = np.array(row["preds"])[mask]
                per_alpha[a] = {"auroc": safe_auroc(g_labels, scores),
                                "accuracy": float((preds == g_labels).mean())}
                long_rows.append({"trajectory": traj, "adapter": tag, "group": g, "n": n,
                                  "alpha": a, **per_alpha[a]})
            native = per_alpha[0.0]["auroc"]
            plateau_vals = [per_alpha[a]["auroc"] for a in PLATEAU_ALPHAS if a in per_alpha]
            plateau_min = float(np.nanmin(plateau_vals)) if plateau_vals else np.nan
            final = per_alpha[1.0]["auroc"]
            struct_rows.append({
                "trajectory": traj, "adapter": tag, "group": g, "n": n,
                "native_auroc": native,
                "plateau_min_auroc": plateau_min,
                "plateau_mean_auroc": float(np.nanmean(plateau_vals)) if plateau_vals else np.nan,
                "alpha1_auroc": final,
                "collapse_delta": (native - final) if not (np.isnan(native) or np.isnan(final)) else np.nan,
                "plateau_holds": bool(plateau_min >= native - PLATEAU_TOL) if not np.isnan(plateau_min) else None,
                "collapses_at_1": bool(final <= COLLAPSE_AUROC) if not np.isnan(final) else None,
                "native_acc": per_alpha[0.0]["accuracy"],
                "alpha1_acc": per_alpha[1.0]["accuracy"],
            })
            for a in KEY_ALPHAS:
                scores = np.array(sweep[a]["scores"])[mask]
                lo, hi = boot_auroc_ci(g_labels, scores)
                ci_rows.append({"trajectory": traj, "adapter": tag, "group": g, "n": n,
                                "alpha": a, "auroc": safe_auroc(g_labels, scores),
                                "ci_lo": lo, "ci_hi": hi})

    long_df, struct_df, ci_df = pd.DataFrame(long_rows), pd.DataFrame(struct_rows), pd.DataFrame(ci_rows)
    table_a = struct_df[["trajectory", "adapter", "group", "n", "native_auroc", "plateau_min_auroc",
                         "alpha1_auroc", "collapse_delta", "native_acc", "alpha1_acc",
                         ]].sort_values(["trajectory", "group", "adapter"]).round(4)
    agg = struct_df.groupby(["trajectory", "group"]).agg(
        n=("n", "first"), runs=("adapter", "count"),
        native_auroc_mean=("native_auroc", "mean"), native_auroc_sd=("native_auroc", "std"),
        plateau_min_mean=("plateau_min_auroc", "mean"), plateau_min_sd=("plateau_min_auroc", "std"),
        alpha1_auroc_mean=("alpha1_auroc", "mean"), alpha1_auroc_sd=("alpha1_auroc", "std"),
        collapse_delta_mean=("collapse_delta", "mean"),
    ).reset_index().round(4)

    def _count(s):
        vals = [v for v in s if v is not None]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}"

    table_c = struct_df.groupby("group").agg(
        runs=("adapter", "count"), plateau_holds=("plateau_holds", _count),
        collapses_at_alpha1=("collapses_at_1", _count)).reset_index()
    return {"A": table_a, "B": agg, "C": table_c, "D": ci_df.round(4), "long": long_df.round(4)}


def _synthetic_sweeps(root, rng, n=160):
    """12 payloads under the real pinned pair_ids, with per-item scores and metadata."""
    kinds = [("simple", "affirmed"), ("simple", "negated"), ("conjunction", "affirmed"),
             ("disjunction", "affirmed"), ("disjunction", "negated")]
    for pid in residuals.run_ids(residuals.TRAINED_VARIANTS):
        parsed = naming.parse_sweep_dir(pid)
        labels = rng.integers(0, 2, n)
        items = []
        for lab in labels:
            st, pol = kinds[rng.integers(len(kinds))]
            items.append({"label": int(lab), "statement_type": st, "polarity": pol, "topic": "t"})
        sweep = []
        for i in range(11):
            s = 1 / (1 + np.exp(-(labels - 0.5) * (2 - 1.5 * (i / 10) ** 2) - rng.normal(0, 1, n)))
            sweep.append({"alpha": round(i * 0.1, 1), "scores": s.tolist(),
                          "preds": (s > 0.5).astype(int).tolist()})
        d = Path(root) / pid
        d.mkdir(parents=True)
        json.dump({"pair_id": pid, "variant": parsed["variant"],
                   "display_label": naming.display_pair_label(
                       parsed["src_id"], parsed["src_layer"],
                       parsed["tgt_id"], parsed["tgt_layer"]),
                   "items": items, "alpha_sweep": sweep}, open(d / "sweep_results.json", "w"))


def test_statement_tables_match_original_code(tmp_path):
    _synthetic_sweeps(tmp_path, np.random.default_rng(5))
    ids = sorted(residuals.run_ids(residuals.TRAINED_VARIANTS))
    want = _original_cell10([json.load(open(tmp_path / p / "sweep_results.json")) for p in ids],
                            N_BOOT=N_BOOT_OFFLINE)
    got = residuals.statement_type_tables(tmp_path, n_boot=N_BOOT_OFFLINE)
    for key in residuals.STATEMENT_TABLES:
        pd.testing.assert_frame_equal(got[key].reset_index(drop=True),
                                      want[key].reset_index(drop=True), check_exact=True)


def test_statement_tables_order_matters(tmp_path):
    """Guard for the population pin: Table D changes if the run order changes."""
    _synthetic_sweeps(tmp_path, np.random.default_rng(5))
    ids = sorted(residuals.run_ids(residuals.TRAINED_VARIANTS))
    ref = residuals.statement_type_tables(tmp_path, n_boot=N_BOOT_OFFLINE)["D"]
    rev = _original_cell10([json.load(open(tmp_path / p / "sweep_results.json"))
                            for p in reversed(ids)], N_BOOT=N_BOOT_OFFLINE)["D"]
    assert not ref.sort_values(["trajectory", "adapter", "group", "alpha"]).reset_index(drop=True) \
        .equals(rev.sort_values(["trajectory", "adapter", "group", "alpha"]).reset_index(drop=True))


# --- 2. published regression --------------------------------------------------

@pytest.fixture(scope="module")
def sweeps_root():
    local = os.environ.get("SHARED_TRUTH_RESULTS_DIR")
    if local:
        return Path(local) / storage.RESULTS_PREFIX
    try:
        return Path(storage.fetch_results(include_analysis=True)) / storage.RESULTS_PREFIX
    except Exception as e:  # network, auth, proxy
        pytest.skip(f"results repo unreachable: {type(e).__name__}: {e}")


def test_statement_tables_reproduce_published(sweeps_root):
    tables = residuals.statement_type_tables(sweeps_root)
    problems = residuals.compare_statement_tables(
        tables, sweeps_root / "analysis_statement_types")
    bad = [f"{k}: {p}" for k, ps in problems.items() for p in ps]
    assert not bad, "statement-type tables changed:\n" + "\n".join(bad)