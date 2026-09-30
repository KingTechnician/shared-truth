"""Rebuttal tables: A-D (statement types), E (baselines), F (residual geometry).
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
    "TRAINED_VARIANTS", "SIMPLE_TRAINED_VARIANTS", "BASELINE_VARIANTS", "run_ids",
    "REPROBE_TOL", "TRAINED_MAPS", "check_reprobe_claims",
    "STATEMENT_TABLES", "statement_type_tables", "compare_statement_tables",
]

# The three multi-seed trajectories (base adapter repos). Seeds, baselines and
# dumps are all keyed off these.
TRAJECTORY_ADAPTERS = [
    "KingTechnician/gemma_2_2b_instruct_l13_to_llama_3.2_3b_instruct_l12",
    "KingTechnician/llama_3.2_3b_instruct_l12_to_gemma_2_2b_instruct_l13",
    "KingTechnician/gemma_2_2b_instruct_l13_to_qwen_2.5_1.5b_instruct_l17",
]

# The same 12 adapters (orig + seeds 42-44 on 3 trajectories) were swept twice:
#   SIMPLE_TRAINED_VARIANTS  simple statements (n=1599), the eval behind the
#                            rebuttal's multi-seed claims and Table 7
#   TRAINED_VARIANTS         full test set (n=2684), for the statement-type
#                            analysis, the baselines, and figures R1/R2
# The 6 Mse8 baseline runs (BASELINE_VARIANTS) exist on the full test set only.
SIMPLE_TRAINED_VARIANTS = ["truth_origadapter", "truth_seed42", "truth_seed43", "truth_seed44"]
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

# The re-probe is not bit-reproducible. A rerun differed from
# the published values by up to 0.0059 on 8 of 21 rows, with every other
# column exact. Only 1 of 21 fits hit lbfgs's limit, so non-convergence is not
# the main cause. Likely (unverified): at C=1e6 on near-separable activations
# the loss is nearly flat along the separating directions, so where lbfgs
# meets its tolerance depends on the numerical environment (numpy/scipy/BLAS).
# Rows record reprobe_converged.
REPROBE_TOL = 0.01

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
    """Held-out AUROC of a fresh probe, and whether lbfgs actually converged."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    lr = LogisticRegression(C=REPROBE_C, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(X[tr], labels[tr])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return float(roc_auc_score(labels[te], lr.decision_function(X[te]))), converged


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
        reprobe, converged = _reprobe_auroc(M, labels, tr, te, REPROBE_MAX_ITER)
        rows.append({
            "trajectory": npz_path.stem, "map": key.replace("map_", ""),
            "gap_ratio_tG": truth_gap(M) / gap_nat_G if gap_nat_G else np.nan,
            "polgap_ratio_tP": (pol_gap(M) / gap_nat_P
                                if gap_nat_P and not np.isnan(gap_nat_P) else np.nan),
            "resid_aniso_tG": aniso,
            "aniso_between": between,
            "aniso_within": aniso - between,
            "auroc_tG_read": float(roc_auc_score(labels[te], (M @ tG)[te])),
            "auroc_reprobe": reprobe,
            "reprobe_converged": converged,
        })

    reprobe, converged = _reprobe_auroc(H_T, labels, tr, te, NATIVE_REPROBE_MAX_ITER)
    rows.append({
        "trajectory": npz_path.stem, "map": NATIVE_LABEL,
        "gap_ratio_tG": 1.0, "polgap_ratio_tP": 1.0,
        "resid_aniso_tG": np.nan, "aniso_between": np.nan, "aniso_within": np.nan,
        "auroc_tG_read": float(roc_auc_score(labels[te], (H_T @ tG)[te])),
        "auroc_reprobe": reprobe,
        "reprobe_converged": converged,
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


def compare_published(rows, csv_path, table, reprobe_tol=REPROBE_TOL):
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


TRAINED_MAPS = ["orig", "seed42", "seed43", "seed44"]


def check_reprobe_claims(rows):
    """The rebuttal's rotation-diagnostic claims, checked on Table F rows.

    Returns {claim: (holds, detail)}. These are what the text asserts, so they
    are what must survive an environment change, unlike the 4th decimal.
    """
    def col(maps, c):
        return [r[c] for r in rows if r["map"] in maps]

    tr_rp, tr_read = col(TRAINED_MAPS, "auroc_reprobe"), col(TRAINED_MAPS, "auroc_tG_read")
    tr_gap = col(TRAINED_MAPS, "gap_ratio_tG")
    shuf_rp = col(["shuf_out"], "auroc_reprobe")
    return {
        "trained re-probe range is 0.84-0.95 at 2 dp":
            ((round(min(tr_rp), 2), round(max(tr_rp), 2)) == (0.84, 0.95),
             f"{min(tr_rp):.4f}-{max(tr_rp):.4f}"),
        "trained gap ratio along t_G is 0.02-0.11 at 2 dp":
            ((round(min(tr_gap), 2), round(max(tr_gap), 2)) == (0.02, 0.11),
             f"{min(tr_gap):.4f}-{max(tr_gap):.4f}"),
        "re-probe beats the t_G read on every trained map":
            (all(a > b for a, b in zip(tr_rp, tr_read)),
             f"min margin {min(a - b for a, b in zip(tr_rp, tr_read)):+.4f}"),
        "shuffled control re-probes near chance (< 0.60)":
            (max(shuf_rp) < 0.60, f"max {max(shuf_rp):.4f}"),
    }


# --- Tables A-D: statement types ---------------------------------------------
# The published CSVs were computed over the 12 full-test-set trained runs only (the baselines came
# later), in sorted pair_id order, with one RNG seeded 0 and 1000 resamples.

PLATEAU_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
KEY_ALPHAS = [0.0, 0.7, 1.0]
PLATEAU_TOL = 0.02      # plateau "holds" if min AUROC over alpha<=0.7 >= native - tol
COLLAPSE_AUROC = 0.60   # collapse "occurs" if AUROC at alpha=1 <= this
STATEMENT_MIN_N = 20
STATEMENT_N_BOOT = 1000

# key -> published CSV stem under sweep-results/analysis_statement_types/
STATEMENT_TABLES = {
    "A": "tableA_per_run_detail",
    "B": "tableB_trajectory_aggregate",
    "C": "tableC_structural_consistency",
    "D": "tableD_bootstrap_cis",
    "long": "long_form_all_metrics",
}


def statement_type_tables(sweeps_root, ids=None, seed=0, n_boot=STATEMENT_N_BOOT):
    """Tables A-D and the long-form table, as DataFrames rounded as published.

    ids defaults to the 12 full-test-set trained runs. They are processed in
    sorted order.
    """
    import json
    import pandas as pd

    ids = sorted(ids if ids is not None else run_ids(TRAINED_VARIANTS))
    rng = np.random.default_rng(seed)
    long_rows, ci_rows, struct_rows = [], [], []

    for pid in ids:
        pl = json.load(open(Path(sweeps_root) / pid / "sweep_results.json"))
        traj, tag = pl["display_label"], naming.adapter_tag(pl["variant"])
        items = pl["items"]
        labels = np.array([it["label"] for it in items])
        groups = np.array([metrics.statement_group(it) for it in items])
        sweep = {round(r["alpha"], 1): r for r in pl["alpha_sweep"]}

        for g in ["all"] + sorted(set(groups)):
            mask = np.ones(len(items), bool) if g == "all" else (groups == g)
            n = int(mask.sum())
            if n < STATEMENT_MIN_N:
                continue
            g_labels = labels[mask]

            per_alpha = {}
            for a, row in sweep.items():
                preds = np.array(row["preds"])[mask]
                per_alpha[a] = {
                    "auroc": metrics.safe_auroc(g_labels, np.array(row["scores"])[mask],
                                                min_n=STATEMENT_MIN_N),
                    "accuracy": float((preds == g_labels).mean()),
                }
                long_rows.append({"trajectory": traj, "adapter": tag, "group": g, "n": n,
                                  "alpha": a, **per_alpha[a]})

            native = per_alpha[0.0]["auroc"]
            plateau = [per_alpha[a]["auroc"] for a in PLATEAU_ALPHAS if a in per_alpha]
            plateau_min = float(np.nanmin(plateau)) if plateau else np.nan
            final = per_alpha[1.0]["auroc"]
            struct_rows.append({
                "trajectory": traj, "adapter": tag, "group": g, "n": n,
                "native_auroc": native,
                "plateau_min_auroc": plateau_min,
                "plateau_mean_auroc": float(np.nanmean(plateau)) if plateau else np.nan,
                "alpha1_auroc": final,
                "collapse_delta": (native - final)
                                  if not (np.isnan(native) or np.isnan(final)) else np.nan,
                "plateau_holds": (bool(plateau_min >= native - PLATEAU_TOL)
                                  if not np.isnan(plateau_min) else None),
                "collapses_at_1": bool(final <= COLLAPSE_AUROC) if not np.isnan(final) else None,
                "native_acc": per_alpha[0.0]["accuracy"],
                "alpha1_acc": per_alpha[1.0]["accuracy"],
            })

            for a in KEY_ALPHAS:
                scores = np.array(sweep[a]["scores"])[mask]
                lo, hi = metrics.boot_auroc_ci(g_labels, scores, rng, n_boot=n_boot,
                                               min_n=STATEMENT_MIN_N)
                ci_rows.append({"trajectory": traj, "adapter": tag, "group": g, "n": n,
                                "alpha": a,
                                "auroc": metrics.safe_auroc(g_labels, scores,
                                                            min_n=STATEMENT_MIN_N),
                                "ci_lo": lo, "ci_hi": hi})

    long_df, struct_df, ci_df = (pd.DataFrame(r) for r in (long_rows, struct_rows, ci_rows))

    table_a = struct_df[[
        "trajectory", "adapter", "group", "n",
        "native_auroc", "plateau_min_auroc", "alpha1_auroc", "collapse_delta",
        "native_acc", "alpha1_acc",
    ]].sort_values(["trajectory", "group", "adapter"]).round(4)

    table_b = struct_df.groupby(["trajectory", "group"]).agg(
        n=("n", "first"),
        runs=("adapter", "count"),
        native_auroc_mean=("native_auroc", "mean"),
        native_auroc_sd=("native_auroc", "std"),
        plateau_min_mean=("plateau_min_auroc", "mean"),
        plateau_min_sd=("plateau_min_auroc", "std"),
        alpha1_auroc_mean=("alpha1_auroc", "mean"),
        alpha1_auroc_sd=("alpha1_auroc", "std"),
        collapse_delta_mean=("collapse_delta", "mean"),
    ).reset_index().round(4)

    def _count(col):
        vals = [v for v in col if v is not None]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}"

    table_c = struct_df.groupby("group").agg(
        runs=("adapter", "count"),
        plateau_holds=("plateau_holds", _count),
        collapses_at_alpha1=("collapses_at_1", _count),
    ).reset_index()

    return {"A": table_a, "B": table_b, "C": table_c,
            "D": ci_df.round(4), "long": long_df.round(4)}


def compare_statement_tables(tables, analysis_dir):
    """Diff tables against the published CSVs; returns {key: [problems]}.

    Compared as CSV text round-trips, so a match means the published file
    would be rewritten byte for byte in content.
    """
    import io
    import pandas as pd

    out = {}
    for key, df in tables.items():
        want = pd.read_csv(Path(analysis_dir) / f"{STATEMENT_TABLES[key]}.csv")
        got = pd.read_csv(io.StringIO(df.to_csv(index=False)))
        probs = []
        if list(got.columns) != list(want.columns):
            probs.append(f"columns differ: {list(got.columns)} vs {list(want.columns)}")
        elif got.shape != want.shape:
            probs.append(f"shape differs: {got.shape} vs {want.shape}")
        else:
            got, want = got.reset_index(drop=True), want.reset_index(drop=True)
            for col in got.columns:
                a, b = got[col], want[col]
                if a.dtype.kind in "fi" and b.dtype.kind in "fi":
                    diff = ~((a == b) | (a.isna() & b.isna()))
                else:
                    diff = a.astype(str) != b.astype(str)
                for i in np.flatnonzero(diff.to_numpy())[:5]:
                    va, vb = a[i], b[i]
                    va, vb = (va.item() if hasattr(va, "item") else va,
                              vb.item() if hasattr(vb, "item") else vb)
                    probs.append(f"row {i} {col}: computed {va!r}, published {vb!r}")
                if diff.sum() > 5:
                    probs.append(f"... {int(diff.sum()) - 5} more in {col}")
        out[key] = probs
    return out