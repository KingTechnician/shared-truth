"""Regression tests for shared_truth.residuals (rebuttal Tables E and F).

Two layers:

1. Offline equivalence (always runs, no network): the module is checked
   against a verbatim copy of the original notebook code (experiment
   notebook, Cell 12) on synthetic dumps and sweeps. This pins the port
   itself, including the RNG consumption order.

2. Published regression (needs the results repo): recomputes both tables from
   the real sweeps and activation dumps and compares them with the CSVs the
   rebuttal numbers came from:
     sweep-results/analysis_residuals/tableE_baseline_vs_trained.csv
     sweep-results/analysis_residuals/tableF_residual_geometry.csv

       HF_TOKEN=... pytest tests/test_residuals_regression.py -v

   Pulls ~280 MB of dumps on first run. Set SHARED_TRUTH_RESULTS_DIR to an
   existing snapshot to skip the download. Skips if the Hub is unreachable.


"""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from shared_truth import residuals, storage

REPROBE_TOL = float(os.environ.get("SHARED_TRUTH_REPROBE_TOL", residuals.REPROBE_TOL))
ANALYSIS_DIR = "analysis_residuals"


# --- helpers ----------------------------------------------------------------

def _num(v):
    if v is None or v == "":
        return math.nan
    return float(v)


def _same(a, b, tol=0.0):
    a, b = _num(a), _num(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol + 1e-12


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compare(got_rows, want_rows, key_cols, tolerances):
    got = {tuple(r[k] for k in key_cols): r for r in got_rows}
    want = {tuple(r[k] for k in key_cols): r for r in want_rows}
    problems = []
    if set(got) != set(want):
        problems.append(f"row keys differ: only computed {sorted(set(got) - set(want))}, "
                        f"only published {sorted(set(want) - set(got))}")
    for k in sorted(set(got) & set(want)):
        for col, tol in tolerances.items():
            if not _same(got[k][col], want[k][col], tol):
                problems.append(f"{k} {col}: computed {got[k][col]!r}, published {want[k][col]!r}")
    return problems


# --- 1. offline equivalence against the original notebook code --------------

def _original_table_f(dump_dir):
    """Verbatim logic of the experiment notebook's Cell 12, Table F section."""
    RNG = np.random.default_rng(0)

    def unit(v):
        v = np.asarray(v, np.float64).ravel()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    rows_f = []
    for npz_path in sorted(Path(dump_dir).glob("*.npz")):
        d = np.load(npz_path, allow_pickle=False)
        H_T = d["h_T"].astype(np.float64)
        labels = d["labels"].astype(int)
        polarity = d["polarity"]
        tG, tP = unit(d["t_G"]), unit(d["t_P"])
        if tG.shape[0] != H_T.shape[1]:
            continue
        mT, mF = labels == 1, labels == 0
        aff, neg = polarity == "affirmed", polarity == "negated"

        def truth_gap(X):
            return float(X[mT].mean(0) @ tG - X[mF].mean(0) @ tG)

        def pol_gap(X):
            if aff.sum() < 20 or neg.sum() < 20:
                return np.nan
            return float(X[aff].mean(0) @ tP - X[neg].mean(0) @ tP)

        gap_nat_G, gap_nat_P = truth_gap(H_T), pol_gap(H_T)
        idx = np.arange(len(labels))
        tr, te = [], []
        for cls in (0, 1):
            c = idx[labels == cls]; c = c[RNG.permutation(len(c))]
            half = len(c) // 2
            tr.append(c[:half]); te.append(c[half:])
        tr, te = np.concatenate(tr), np.concatenate(te)

        for key in ["map_orig", "map_seed42", "map_seed43", "map_seed44",
                    "rand_out", "shuf_out"]:
            if key not in d.files: continue
            M = d[key].astype(np.float64)
            R = M - H_T
            proj = R @ tG
            per_dim_var = R.var(axis=0).mean()
            aniso = float(proj.var() / max(per_dim_var, 1e-12))
            between = float((proj[mT].mean() - proj[mF].mean()) ** 2 / 4 / max(per_dim_var, 1e-12))
            within = aniso - between
            dir_read = M @ tG
            auroc_dir = float(roc_auc_score(labels[te], dir_read[te]))
            lr = LogisticRegression(C=1e6, max_iter=20000)
            lr.fit(M[tr], labels[tr])
            auroc_reprobe = float(roc_auc_score(labels[te], lr.decision_function(M[te])))
            rows_f.append({
                "trajectory": npz_path.stem, "map": key.replace("map_", ""),
                "gap_ratio_tG": round(truth_gap(M) / gap_nat_G, 3) if gap_nat_G else np.nan,
                "polgap_ratio_tP": round(pol_gap(M) / gap_nat_P, 3) if gap_nat_P and not np.isnan(gap_nat_P) else np.nan,
                "resid_aniso_tG": round(aniso, 2),
                "aniso_between": round(between, 3),
                "aniso_within": round(within, 2),
                "auroc_tG_read": round(auroc_dir, 4),
                "auroc_reprobe": round(auroc_reprobe, 4),
            })
        auroc_nat_dir = float(roc_auc_score(labels[te], (H_T @ tG)[te]))
        lr = LogisticRegression(C=1e6, max_iter=2000)
        lr.fit(H_T[tr], labels[tr])
        rows_f.append({
            "trajectory": npz_path.stem, "map": "NATIVE (ref)",
            "gap_ratio_tG": 1.0, "polgap_ratio_tP": 1.0,
            "resid_aniso_tG": np.nan, "aniso_between": np.nan, "aniso_within": np.nan,
            "auroc_tG_read": round(auroc_nat_dir, 4),
            "auroc_reprobe": round(float(roc_auc_score(labels[te], lr.decision_function(H_T[te]))), 4),
        })
    return rows_f


def _synthetic_dump(path, rng, n=240, dim=24):
    labels = rng.integers(0, 2, n)
    polarity = np.where(rng.random(n) < 0.6, "affirmed", "negated")
    tG, tP = rng.normal(size=dim), rng.normal(size=dim)
    H_T = rng.normal(size=(n, dim)) + np.outer(labels - 0.5, tG)
    maps = {k: (0.3 * H_T + rng.normal(size=(n, dim))).astype(np.float16)
            for k in residuals.MAP_KEYS}
    np.savez_compressed(path, h_S=rng.normal(size=(n, dim)).astype(np.float16),
                        h_T=H_T.astype(np.float16), **maps,
                        t_G=tG.astype(np.float32), t_P=tP.astype(np.float32),
                        labels=labels.astype(np.int8), polarity=polarity,
                        statement_type=np.array(["simple"] * n))


def test_table_f_matches_original_code(tmp_path):
    rng = np.random.default_rng(123)
    # Three dumps, so a wrong RNG hand-off between dumps would show up.
    for name in ("b_traj", "a_traj", "c_traj"):
        _synthetic_dump(tmp_path / f"{name}.npz", rng)
    want = _original_table_f(tmp_path)
    got = residuals.round_published(residuals.table_f(tmp_path.glob("*.npz")), "F")
    cols = {c: 0.0 for c in residuals.TABLE_F_ROUND}
    assert not _compare(got, want, ("trajectory", "map"), cols)
    assert [(r["trajectory"], r["map"]) for r in got] == \
           [(r["trajectory"], r["map"]) for r in want]


def test_table_e_matches_original_helpers():
    """strict overshoot / tau0 delta via metrics vs the notebook's own helpers."""
    rng = np.random.default_rng(7)

    def orig_opt_acc(t, f):
        scores = np.concatenate([t, f])
        labels = np.concatenate([np.ones(len(t)), np.zeros(len(f))])[np.argsort(scores)]
        n = len(labels); cum = np.cumsum(labels); tot = cum[-1]
        return float(max(((tot - cum) + (np.arange(n) + 1 - cum)).max(), tot, n - tot) / n)

    payloads = {}
    for base in residuals.TRAJECTORY_ADAPTERS:
        src, sl, tgt, tl = residuals.naming.parse_adapter_repo(base)
        for variant, _ in residuals.TABLE_E_MAPS:
            sweep = []
            for a in range(11):
                t, f = rng.random(150) + 0.02 * a, rng.random(130)
                sweep.append({"alpha": a / 10, "accuracy": float(rng.random()),
                              "auroc": float(rng.random()),
                              "true_scores": t.tolist(), "false_scores": f.tolist()})
            payloads[residuals.naming.pair_id(src, sl, tgt, tl, variant)] = {"alpha_sweep": sweep}

    rows = residuals.table_e(None, load=payloads.get)
    for r in rows:
        sweep = payloads[r["pair_id"]]["alpha_sweep"]
        accs = [orig_opt_acc(np.array(x["true_scores"]), np.array(x["false_scores"])) for x in sweep]
        assert r["strict_dAcc_taustar_pp"] == pytest.approx((max(accs) - accs[0]) * 100, abs=1e-12)
        tau0 = [x["accuracy"] for x in sweep]
        assert r["peak_dAcc_tau0_pp"] == pytest.approx((max(tau0) - tau0[0]) * 100, abs=1e-12)


# --- 2. published regression -------------------------------------------------

@pytest.fixture(scope="module")
def results_root():
    local = os.environ.get("SHARED_TRUTH_RESULTS_DIR")
    if local:
        return Path(local)
    try:
        return Path(storage.fetch_results(include_dumps=True, include_analysis=True))
    except Exception as e:  # network, auth, proxy
        pytest.skip(f"results repo unreachable: {type(e).__name__}: {e}")


def test_table_e_reproduces_published(results_root):
    sweeps = results_root / storage.RESULTS_PREFIX
    want = _read_csv(sweeps / ANALYSIS_DIR / "tableE_baseline_vs_trained.csv")
    got = residuals.round_published(residuals.table_e(sweeps), "E")
    cols = {c: 0.0 for c in residuals.TABLE_E_ROUND}
    problems = _compare(got, want, ("trajectory", "map"), cols)
    assert not problems, "Table E changed:\n" + "\n".join(problems)


def test_table_f_reproduces_published(results_root):
    sweeps = results_root / storage.RESULTS_PREFIX
    dumps = sorted((sweeps / storage.DUMPS_DIR).glob("*.npz"))
    assert len(dumps) == 3, f"expected 3 activation dumps, found {len(dumps)}"
    want = _read_csv(sweeps / ANALYSIS_DIR / "tableF_residual_geometry.csv")
    got = residuals.round_published(residuals.table_f(dumps), "F")
    cols = {c: 0.0 for c in residuals.TABLE_F_ROUND}
    cols["auroc_reprobe"] = REPROBE_TOL
    problems = _compare(got, want, ("trajectory", "map"), cols)
    assert not problems, "Table F changed:\n" + "\n".join(problems)