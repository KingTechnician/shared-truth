"""Truth-probe loading, projection onto (t_G, t_P), and scoring.


"""

import json
import warnings

import numpy as np
import sklearn
import skops.io as sio
from huggingface_hub import hf_hub_download

from .env import recorded_sklearn_version

__all__ = ["load_probe", "project", "score_one", "score_batch"]


def _check_sklearn_version(repo_id, model_path):
    """Refuse to load a probe under a different scikit-learn than saved it.

    A mismatch doesn't always fail loudly. Across 1.8 it crashes in
    predict_proba ('no attribute multi_class'), but other version gaps can load
    fine and score differently. Fail at load time, with the fix in the message.
    """
    saved = recorded_sklearn_version(model_path)
    if saved and saved != sklearn.__version__:
        raise RuntimeError(
            f"probe {repo_id} was saved with scikit-learn {saved}, but this runtime has "
            f"{sklearn.__version__}. Run shared_truth.env.pin_sklearn('{saved}') at the top "
            f"of a fresh session (or pip install scikit-learn=={saved} and restart)."
        )


def load_probe(repo_id, strict=True):
    """Load a logistic probe and its (t_G, t_P) truth directions.

    Returns (sklearn_estimator, (t_G, t_P)).

    strict=True (default, from the Procrustes notebook): a missing or malformed
    truth_directions.json raises.

    strict=False (from the adapter notebook): returns directions=None instead,
    which makes `project` a no-op and feeds full-dimensional activations to a
    probe fitted on 2-D features. That path produces plausible-looking wrong
    numbers with no error, so it warns loudly and exists only to reproduce
    artifacts generated before this was fixed. Do not use it for new runs.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename="model.skops")
    _check_sklearn_version(repo_id, model_path)
    untrusted = sio.get_untrusted_types(file=model_path)
    lr = sio.load(model_path, trusted=untrusted)

    try:
        td_path = hf_hub_download(repo_id=repo_id, filename="truth_directions.json")
        with open(td_path) as f:
            td = json.load(f)
        directions = (
            np.array(td["t_G"], dtype=np.float32).flatten(),
            np.array(td["t_P"], dtype=np.float32).flatten(),
        )
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"could not load truth_directions.json from {repo_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        warnings.warn(
            f"load_probe({repo_id}, strict=False): truth directions unavailable "
            f"({type(exc).__name__}). Activations will be passed to the probe "
            f"UNPROJECTED. Any metric computed from this probe is invalid.",
            RuntimeWarning, stacklevel=2,
        )
        directions = None

    return lr, directions


def project(act, directions):
    """Project activations onto the two truth directions.

    act: (d,) or (n, d). Returns (n, 2). Pass-through when directions is None
    (see load_probe strict=False).
    """
    if directions is None:
        return act
    t_G, t_P = directions
    return np.column_stack((np.dot(act, t_G), np.dot(act, t_P)))


def score_one(probe, directions, act):
    """Single-activation scoring. Returns (pred:int, p_true:float).

    Kept because the per-item loop in native_ceilings/alpha_sweep depends on
    the exact ordering of predict/predict_proba calls that produced the
    published numbers.
    """
    feats = project(act, directions)
    return int(probe.predict(feats)[0]), float(probe.predict_proba(feats)[0][1])


def score_batch(probe, directions, acts):
    """Vectorized scoring over (n, d) activations. Returns (preds, p_true).

    Numerically identical to looping score_one for a linear probe, and far
    faster. Used by the Procrustes path, which never needed per-item ordering.
    """
    feats = project(np.atleast_2d(acts), directions)
    return probe.predict(feats).astype(int), probe.predict_proba(feats)[:, 1]