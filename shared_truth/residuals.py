"""Rebuttal Tables E and F: baseline controls and residual geometry.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import metrics, naming

__all__ = [
    "TRAJECTORY_ADAPTERS", "TABLE_E_MAPS", "MAP_KEYS", "NATIVE_LABEL",
    "REPROBE_C", "REPROBE_MAX_ITER", "NATIVE_REPROBE_MAX_ITER", "MIN_POLARITY_N",
    "TABLE_E_ROUND", "TABLE_F_ROUND",
    "trajectory_slug", "table_e", "split_half", "residual_geometry", "table_f",
    "round_published", "compare_published",
    "TRAINED_VARIANTS", "BASELINE_VARIANTS", "run_ids",
]

# The three multi-seed trajectories (base adapter repos). Seeds, baselines and
# dumps are all keyed off these.
TRAJECTORY_ADAPTERS = [
    "KingTechnician/gemma_2_2b_instruct_l13_to_llama_3.2_3b_instruct_l12",
    "KingTechnician/llama_3.2_3b_instruct_l12_to_gemma_2_2b_instruct_l13",
    "KingTechnician/gemma_2_2b_instruct_l13_to_qwen_2.5_1.5b_instruct_l17",
]

# The 12 multi-seed runs are TRAINED_VARIANTS on each trajectory; the 6 Mse8
# baseline runs are BASELINE_VARIANTS. All on the full test set (n=2684).
TRAINED_VARIANTS = ["stmt_origadapter", "stmt_seed42", "stmt_seed43", "stmt_seed44"]
BASELINE_VARIANTS = ["stmt_randbase", "stmt_shufbase"]

TABLE_E_MAPS = [
    ("stmt_origadapter", "trained (orig)"),
    ("stmt_randbase", "random-weight"),
    ("stmt_shufbase", "shuffled"),
]

# npz keys, in the order rows are emitted. Output labels strip "map_", which
# is what the published CSV shows (orig, seed42, ..., rand_out, shuf_out).
MAP_KEYS = ["map_orig", "map_seed42", "map_seed43", "map_seed44", "rand_out", "shuf_out"]
NATIVE_LABEL = "NATIVE (ref)"

REPROBE_C = 1e6
REPROBE_MAX_ITER = 20000
# The native reference row was fit with max_iter=2000 in the original, not
# 20000. Kept as-is so the published row reproduces.
NATIVE_REPROBE_MAX_ITER = 2000
MIN_POLARITY_N = 20

# Rounding used in the published CSVs (tableE_baseline_vs_trained.csv,
# tableF_residual_geometry.csv).
TABLE_E_ROUND = {"native_acc": 2, "peak_dAcc_tau0_pp": 2, "strict_dAcc_taustar_pp": 2,
                 "auroc_a0": 4, "auroc_a1": 4}
TABLE_F_ROUND = {"gap_ratio_tG": 3, "polgap_ratio_tP": 3, "resid_aniso_tG": 2,
                 "aniso_between": 3, "aniso_within": 2,
                 "auroc_tG_read": 4, "auroc_reprobe": 4}


def trajectory_slug(base_repo):
    """Adapter repo -> dump stem, e.g. gemma2-2b-instruct_l13_to_llama3_2-3b-instruct_l12."""
    src, sl, tgt, tl = naming.parse_adapter_repo(base_repo)
    return naming.pair_id(src, sl, tgt, tl, "dump").replace("__dump", "")


def run_ids(variants=TRAINED_VARIANTS, bases=TRAJECTORY_ADAPTERS):
    """Pinned pair_ids for the multi-seed trajectories, trajectory-major order."""
    out = []
    for base in bases:
        src, sl, tgt, tl = naming.parse_adapter_repo(base)
        out += [naming.pair_id(src, sl, tgt, tl, v) for v in variants]
    return out


# --- Table E ----------------------------------------------------------------

def table_e(sweeps_root, bases=TRAJECTORY_ADAPTERS, maps=TABLE_E_MAPS, load=None):
    """Baseline vs trained comparison, one row per trajectory x map. Unrounded.

    `sweeps_root` is the sweep-results directory. `load(pid)` can override how a
    payload is read (it must return the sweep_results dict or None).
    """
    import json

    def _load(pid):
        p = Path(sweeps_root) / pid / "sweep_results.json"
        return json.load(open(p)) if p.exists() else None

    load = load or _load
    rows = []
    for base in bases:
        src, sl, tgt, tl = naming.parse_adapter_repo(base)
        label = naming.display_pair_label(src, sl, tgt, tl)
        for variant, name in maps:
            pid = naming.pair_id(src, sl, tgt, tl, variant)
            pl = load(pid)
            if pl is None:
                raise FileNotFoundError(f"missing sweep for {label} / {variant}: {pid}")
            sweep = pl["alpha_sweep"]
            so, _ = metrics.strict_overshoot(sweep)
            rows.append({
                "trajectory": label, "map": name, "pair_id": pid,
                "native_acc": sweep[0]["accuracy"] * 100,
                "peak_dAcc_tau0_pp": metrics.tau0_peak_delta(sweep),
                "strict_dAcc_taustar_pp": so,
                "auroc_a0": sweep[0]["auroc"],
                "auroc_a1": sweep[-1]["auroc"],
            })
    return rows


# --- Table F ----------------------------------------------------------------

def split_half(labels, rng):
    """Label-stratified half split. Consumes rng: one permutation per class, 0 then 1."""
    idx = np.arange(len(labels))
    tr, te = [], []
    for cls in (0, 1):
        c = idx[labels == cls]
        c = c[rng.permutation(len(c))]
        half = len(c) // 2
        tr.append(c[:half])
        te.append(c[half:])
    return np.concatenate(tr), np.concatenate(te)


def _reprobe_auroc(X, labels, tr, te, max_iter):
    lr = LogisticRegression(C=REPROBE_C, max_iter=max_iter)
    lr.fit(X[tr], labels[tr])
    return float(roc_auc_score(labels[te], lr.decision_function(X[te])))


def residual_geometry(npz_path, rng, map_keys=MAP_KEYS):
    """Table F rows for one trajectory's dump. Unrounded.

    Returns [] for a dump saved without truth directions, without touching
    rng, as the original did.
    """
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=False)
    H_T = d["h_T"].astype(np.float64)
    labels = d["labels"].astype(int)
    polarity = d["polarity"]
    tG, tP = metrics.unit(d["t_G"]), metrics.unit(d["t_P"])
    if tG.shape[0] != H_T.shape[1]:
        print(f"{npz_path.name}: no directions stored, skipping")
        return []

    mT, mF = labels == 1, labels == 0
    aff, neg = polarity == "affirmed", polarity == "negated"

    def truth_gap(X):   # (mu_true - mu_false) @ t_G
        return float(X[mT].mean(0) @ tG - X[mF].mean(0) @ tG)

    def pol_gap(X):     # (mu_affirmed - mu_negated) @ t_P
        if aff.sum() < MIN_POLARITY_N or neg.sum() < MIN_POLARITY_N:
            return np.nan
        return float(X[aff].mean(0) @ tP - X[neg].mean(0) @ tP)

    gap_nat_G, gap_nat_P = truth_gap(H_T), pol_gap(H_T)
    tr, te = split_half(labels, rng)

    rows = []
    for key in map_keys:
        if key not in d.files:
            continue
        M = d[key].astype(np.float64)
        R = M - H_T
        proj = R @ tG
        per_dim_var = max(R.var(axis=0).mean(), 1e-12)
        aniso = float(proj.var() / per_dim_var)
        between = float((proj[mT].mean() - proj[mF].mean()) ** 2 / 4 / per_dim_var)
        rows.append({
            "trajectory": npz_path.stem, "map": key.replace("map_", ""),
            "gap_ratio_tG": truth_gap(M) / gap_nat_G if gap_nat_G else np.nan,
            "polgap_ratio_tP": (pol_gap(M) / gap_nat_P
                                if gap_nat_P and not np.isnan(gap_nat_P) else np.nan),
            "resid_aniso_tG": aniso,
            "aniso_between": between,
            "aniso_within": aniso - between,
            "auroc_tG_read": float(roc_auc_score(labels[te], (M @ tG)[te])),
            "auroc_reprobe": _reprobe_auroc(M, labels, tr, te, REPROBE_MAX_ITER),
        })

    rows.append({
        "trajectory": npz_path.stem, "map": NATIVE_LABEL,
        "gap_ratio_tG": 1.0, "polgap_ratio_tP": 1.0,
        "resid_aniso_tG": np.nan, "aniso_between": np.nan, "aniso_within": np.nan,
        "auroc_tG_read": float(roc_auc_score(labels[te], (H_T @ tG)[te])),
        "auroc_reprobe": _reprobe_auroc(H_T, labels, tr, te, NATIVE_REPROBE_MAX_ITER),
    })
    return rows


def table_f(dump_paths, seed=0):
    """Table F over every dump. One RNG across all dumps, in sorted path order.

    Pass the full set of dumps: computing a single trajectory on its own gives
    it a different split than the published table (except for the first).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in sorted(Path(p) for p in dump_paths):
        rows.extend(residual_geometry(p, rng))
    return rows


# --- output -----------------------------------------------------------------

def round_published(rows, table):
    """Round rows the way the published CSV was rounded. table: 'E' or 'F'."""
    spec = {"E": TABLE_E_ROUND, "F": TABLE_F_ROUND}[table]
    out = []
    for r in rows:
        r = dict(r)
        for k, nd in spec.items():
            v = r.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                r[k] = round(float(v), nd)
        out.append(r)
    return out


def compare_published(rows, csv_path, table, reprobe_tol=1e-3):
    """Diff computed rows against a published CSV; returns a list of problems.

    Rows are rounded to the CSV's precision and must match exactly, except
    auroc_reprobe (Table F), which is held to reprobe_tol.
    """
    import csv
    import math

    def num(v):
        return math.nan if v in (None, "") else float(v)

    spec = {"E": TABLE_E_ROUND, "F": TABLE_F_ROUND}[table]
    tol = {c: 0.0 for c in spec}
    if table == "F":
        tol["auroc_reprobe"] = reprobe_tol
    with open(csv_path, newline="", encoding="utf-8") as f:
        want = {(r["trajectory"], r["map"]): r for r in csv.DictReader(f)}
    got = {(r["trajectory"], r["map"]): r for r in round_published(rows, table)}

    problems = []
    if set(got) != set(want):
        problems.append(f"rows differ: only computed {sorted(set(got) - set(want))}, "
                        f"only published {sorted(set(want) - set(got))}")
    for k in sorted(set(got) & set(want)):
        for col, t in tol.items():
            a, b = num(got[k][col]), num(want[k][col])
            same = (math.isnan(a) and math.isnan(b)) or abs(a - b) <= t + 1e-12
            if not same:
                problems.append(f"{k} {col}: computed {a!r}, published {b!r}")
    return problems