"""Regression test: shared_truth.metrics must reproduce the published bootstrap.

Pulls the sweep JSONs and strict_overshoot_bootstrap.json from the results repo
and recomputes all 22 trajectories. CPU-only; takes a few minutes at B=10,000.

    HF_TOKEN=... pytest tests/test_bootstrap_regression.py -v

Guards every change to metrics.optimal_accuracy_batch, optimal_accuracy, and
bootstrap_trajectory. A failure here means a published number would change.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from shared_truth import metrics, storage

FIELDS = ("delta_orig_pp", "ci_low_pp", "ci_high_pp", "median_pp", "n_test", "m1", "m0")
REFERENCE = "strict_overshoot_bootstrap.json"


@pytest.fixture(scope="module")
def data():
    root = Path(storage.fetch_results())
    ref = json.load(open(root / REFERENCE))
    return root, ref


def test_published_bootstrap_reproduces(data):
    root, ref = data
    sweeps = root / storage.RESULTS_PREFIX
    expected = {r["pair_id"]: r for r in ref["per_trajectory"]}

    # Draw order is sorted pair_id over exactly the reference's population.
    rng = np.random.default_rng(ref["seed"])
    mismatches = []
    for pid in sorted(expected):
        sweep = json.load(open(sweeps / pid / "sweep_results.json"))["alpha_sweep"]
        got = metrics.bootstrap_trajectory(sweep, rng, ref["B"])
        want = expected[pid]
        diffs = [k for k in FIELDS if abs(float(got[k]) - float(want[k])) > 1e-9]
        if diffs:
            mismatches.append(f"{pid}: {diffs}")

    assert not mismatches, "published bootstrap changed:\n" + "\n".join(mismatches)
    assert sum(r["ci_low_pp"] > 0 for r in expected.values()) == ref["n_excludes_zero"]


def test_opt_acc_shim_matches_optimal_accuracy():
    """The (true, false) call convention must agree with (scores, labels)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        t, f = rng.random(rng.integers(20, 300)), rng.random(rng.integers(20, 300))
        labels = np.r_[np.ones(len(t)), np.zeros(len(f))].astype(int)
        assert metrics.opt_acc(t, f) == metrics.optimal_accuracy(np.r_[t, f], labels)


def test_tie_guard():
    """Tied scores must not yield an accuracy no threshold can reach."""
    # All scores tied: the only reachable thresholds are all-1 or all-0.
    scores = np.full(10, 0.5)
    labels = np.array([1] * 6 + [0] * 4)
    assert metrics.optimal_accuracy(scores, labels) == pytest.approx(0.6)