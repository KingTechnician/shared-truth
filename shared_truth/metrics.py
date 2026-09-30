"""Metrics.

`optimal_accuracy` is the one function that matters here. Table 7's strict
overshoot point estimates and the bootstrap CIs were computed by two separate
implementations in two separate notebooks (`opt_acc` and
`optimal_accuracy_single`). They agreed on the July 23 reproduction, but only
because continuous predict_proba output never ties. This is the tie-guarded
version.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "optimal_accuracy", "optimal_accuracy_batch", "opt_acc",
    "strict_overshoot", "tau0_peak_delta",
    "safe_auroc", "boot_auroc_ci", "bootstrap_trajectory",
    "statement_group", "unit",
    "DEFAULT_N_BOOT", "DEFAULT_MIN_N", "sweep_diff",
]

DEFAULT_N_BOOT = 2000
DEFAULT_MIN_N = 20


# --- optimal accuracy over thresholds ---------------------------------------

def optimal_accuracy_batch(scores, labels):
    """max_tau accuracy(tau), vectorized over bootstrap replicates.

    scores : (B, n) float — one score array per replicate
    labels : (n,) {0,1} int — shared across replicates
    returns: (B,) float

    Convention: predict 1 iff score > tau. At tau=-inf everything is predicted
    1, so accuracy starts at m1/n; walking tau up flips one sample at a time
    from 1 to 0, changing accuracy by +1/n if that sample's label is 0 and
    -1/n if it is 1.

    Tie handling: a threshold can only sit BETWEEN distinct score values, never
    inside a tied group, since all tied samples must fall on the same side of
    tau. Only cumulative values at positions where sorted_scores[i] is strictly
    less than sorted_scores[i+1] (or the final position) are admissible. With
    continuous probe output ties are effectively impossible; the guard makes
    the function correct for discrete scores too.

    Note: this sweeps one readout direction only, plus the two degenerate
    all-one / all-zero cases. It does not consider a sign-reversed readout.
    That matches the published behavior and is deliberately left unchanged.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    B, n = scores.shape
    starting = int(labels.sum()) / n

    sort_idx = np.argsort(scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, sort_idx, axis=1)
    sorted_labels = labels[sort_idx]

    deltas = np.where(sorted_labels == 0, 1.0, -1.0) / n
    cumulative = np.cumsum(deltas, axis=1)

    valid = np.zeros((B, n), dtype=bool)
    valid[:, -1] = True
    valid[:, :-1] = sorted_scores[:, :-1] < sorted_scores[:, 1:]

    candidates = np.where(valid, starting + cumulative, -np.inf)
    return np.maximum(starting, candidates.max(axis=1))


def optimal_accuracy(scores, labels):
    """Scalar form of optimal_accuracy_batch, for un-resampled data."""
    return float(optimal_accuracy_batch(np.atleast_2d(scores), labels)[0])


def opt_acc(true_scores, false_scores):
    """Compatibility shim for the adapter/figures call convention.

    Those notebooks passed separated true/false score lists rather than
    (scores, labels). Concatenation order is [true..., false...] with labels
    [1..., 0...], matching what the original opt_acc built internally.
    """
    t = np.asarray(true_scores, dtype=np.float64).ravel()
    f = np.asarray(false_scores, dtype=np.float64).ravel()
    scores = np.concatenate([t, f])
    labels = np.concatenate([np.ones(len(t)), np.zeros(len(f))]).astype(int)
    return optimal_accuracy(scores, labels)


# --- overshoot --------------------------------------------------------------

def strict_overshoot(sweep):
    """max_a Acc(tau*(a), a) - Acc(tau*(0), 0), in percentage points.

    Returns (delta_pp, per_alpha_optimal_accuracies). Requires sweep rows to
    carry true_scores/false_scores — Procrustes rows written before this
    package did not.
    """
    accs = [opt_acc(r["true_scores"], r["false_scores"]) for r in sweep]
    return (max(accs) - accs[0]) * 100.0, accs


def tau0_peak_delta(sweep):
    """Peak-minus-baseline using the probe's own fixed threshold, in pp."""
    accs = [r["accuracy"] for r in sweep]
    return (max(accs) - accs[0]) * 100.0


def bootstrap_trajectory(sweep, rng, B):
    """Stratified bootstrap CI for strict overshoot on one trajectory.

    Resampling indices are drawn once and shared across alpha, preserving the
    paired structure: the same resampled items are scored at every alpha, so
    the CI is on the overshoot itself rather than on a difference of
    independently resampled quantities.
    """
    alphas = [r["alpha"] for r in sweep]
    m1 = len(sweep[0]["true_scores"])
    m0 = len(sweep[0]["false_scores"])
    n = m1 + m0
    labels = np.concatenate([np.ones(m1), np.zeros(m0)]).astype(int)

    accs_orig = np.array([
        optimal_accuracy(
            np.concatenate([np.asarray(r["true_scores"]),
                            np.asarray(r["false_scores"])]), labels)
        for r in sweep
    ])
    delta_orig_pp = (accs_orig.max() - accs_orig[0]) * 100

    idx1 = rng.integers(0, m1, size=(B, m1))
    idx0 = rng.integers(0, m0, size=(B, m0))

    accs_per_alpha = np.zeros((B, len(alphas)))
    for k, r in enumerate(sweep):
        ts = np.asarray(r["true_scores"], dtype=np.float64)
        fs = np.asarray(r["false_scores"], dtype=np.float64)
        scores_b = np.concatenate([ts[idx1], fs[idx0]], axis=1)
        accs_per_alpha[:, k] = optimal_accuracy_batch(scores_b, labels)

    delta_pp = (accs_per_alpha.max(axis=1) - accs_per_alpha[:, 0]) * 100
    return {
        "delta_orig_pp": float(delta_orig_pp),
        "ci_low_pp":     float(np.percentile(delta_pp, 2.5)),
        "ci_high_pp":    float(np.percentile(delta_pp, 97.5)),
        "median_pp":     float(np.median(delta_pp)),
        "n_test": n, "m1": m1, "m0": m0,
    }


# --- AUROC ------------------------------------------------------------------

def safe_auroc(labels, scores, min_n=DEFAULT_MIN_N):
    """AUROC, or NaN for slices too small or single-class to be meaningful."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) < min_n or len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def boot_auroc_ci(labels, scores, rng, n_boot=DEFAULT_N_BOOT, min_n=DEFAULT_MIN_N):
    """Percentile bootstrap CI for AUROC. rng is explicit rather than a module
    global so a caller can reproduce a specific run."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) < min_n or len(np.unique(labels)) < 2:
        return (np.nan, np.nan)
    pool = np.arange(len(labels))
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(pool, size=len(pool), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        stats.append(roc_auc_score(labels[idx], scores[idx]))
    if not stats:
        return (np.nan, np.nan)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


# --- slicing / misc ---------------------------------------------------------

def statement_group(item):
    """negated (any type) | simple | conjunction | disjunction | unknown.

    Polarity takes precedence over statement_type, so 'negated' is a group of
    its own rather than a column cutting across the others. This is the
    grouping behind the collapse-gradient result.
    """
    if (item.get("polarity") or "").lower() == "negated":
        return "negated"
    return item.get("statement_type") or "unknown"


def unit(v):
    v = np.asarray(v, np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def sweep_diff(new_sweep, saved_sweep):
    """Per-alpha comparison of a re-run sweep against a saved one.

    Works on both row schemas: true/false scores (every run) and per-item
    preds (Cell 6b onward). items_differing counts changed predictions when
    both sides have preds; otherwise it is the accuracy change in items, a
    lower bound. fp16 matmuls can differ in the last bit across GPUs, so the
    check is stated in items, not float tolerance.
    """
    out = []
    for a, b in zip(new_sweep, saved_sweep, strict=True):
        if round(a["alpha"], 1) != round(b["alpha"], 1):
            raise ValueError(f"alpha mismatch: {a['alpha']} vs {b['alpha']}")
        ta, tb = np.asarray(a["true_scores"]), np.asarray(b["true_scores"])
        fa, fb = np.asarray(a["false_scores"]), np.asarray(b["false_scores"])
        if ta.shape != tb.shape or fa.shape != fb.shape:
            raise ValueError(f"alpha {a['alpha']}: item counts differ "
                             f"({len(ta)}+{len(fa)} vs {len(tb)}+{len(fb)})")
        n = len(ta) + len(fa)
        exact = "preds" in a and "preds" in b
        items = (int((np.asarray(a["preds"]) != np.asarray(b["preds"])).sum()) if exact
                 else int(round(abs(a["accuracy"] - b["accuracy"]) * n)))
        out.append({
            "alpha": a["alpha"], "items_differing": items, "items_exact": exact,
            "max_abs_dscore": float(max(np.abs(ta - tb).max(initial=0),
                                        np.abs(fa - fb).max(initial=0))),
            "abs_dauroc": abs(a["auroc"] - b["auroc"]),
        })
    return out