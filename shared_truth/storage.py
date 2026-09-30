"""Local paths, provenance stamping, and the HF results repo.
"""

import json
import os
import subprocess
from pathlib import Path
 
from huggingface_hub import HfApi, create_repo, snapshot_download
 
__all__ = [
    "SCHEMA_VERSION", "RESULTS_REPO", "EXPECTED_FILES",
    "pair_dir", "is_complete", "load_cached", "save_payload",
    "upload_pair", "ensure_local", "fetch_results", "provenance",
    "RESULTS_PREFIX", "repo_path", "DUMPS_DIR",
    "resolve_revision", "verify_adapter", "AdapterMismatchError",
]

SCHEMA_VERSION = 2   # 1 = pre-rebuttal (no per-item scores); 2 = Cell 6b onward
RESULTS_REPO = "KingTechnician/shared-truth-results"


RESULTS_PREFIX = "sweep-results"
DUMPS_DIR = "activation_dumps"   # under RESULTS_PREFIX; one npz per trajectory
 
 
def repo_path(pid, filename, prefix=RESULTS_PREFIX):
    return f"{prefix}/{pid}/{filename}" if prefix else f"{pid}/{filename}"
 
FORK_PATH = "/content/Closing-Backdoors-Via-Representation-Transfer"
 
EXPECTED_FILES = [
    "sweep_results.json",
    "fig1_alpha_sweep.pdf", "fig1_alpha_sweep.png",
    "fig2_score_distributions.pdf", "fig2_score_distributions.png",
    "fig3_geometry_collapse.pdf", "fig3_geometry_collapse.png",
]
 
 
def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
 
 
def provenance(fork_path=FORK_PATH):
    """Commit SHAs for the code that produced an artifact.
 
    The representation-transfer fork is the one that matters: ModelWrapper's
    hook behavior lives there, and the monkeypatch in activations.py is written
    against a specific version of it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "representation_transfer_sha": _git_sha(fork_path),
        "shared_truth_sha": _git_sha(Path(__file__).resolve().parent.parent),
    }
 
 
def pair_dir(root, pid):
    return Path(root) / pid
 
 
def is_complete(root, pid):
    d = pair_dir(root, pid)
    return all((d / f).exists() for f in EXPECTED_FILES)
 
 
def save_payload(root, pid, payload, fork_path=FORK_PATH, filename="sweep_results.json"):
    """Write a per-pair JSON (sweep_results.json by default) with provenance attached."""
    payload = {**payload, "provenance": provenance(fork_path)}
    out_dir = pair_dir(root, pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
 
 
def load_cached(root, pid, repo_id=RESULTS_REPO, allow_download=True, token=None):
    """Load a pair's payload, pulling it from the results repo if absent locally.
 
    Payloads written before provenance stamping have no 'provenance' key; a
    missing key means schema 1 and no recorded fork SHA.
    """
    path = pair_dir(root, pid) / "sweep_results.json"
    if not path.exists() and allow_download:
        path = Path(ensure_local(pid, "sweep_results.json",
                                 repo_id=repo_id, token=token))
    with open(path) as f:
        return json.load(f)
 
 
def ensure_local(pid, filename, repo_id=RESULTS_REPO, token=None, cache_dir=None):
    """Fetch one artifact from the results repo and return its local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=repo_path(pid, filename),
        token=token or os.environ.get("HF_TOKEN"), cache_dir=cache_dir,
    )
 
 
def fetch_results(pid=None, repo_id=RESULTS_REPO, token=None, local_dir=None,
                  sweeps_only=True, include_dumps=False, include_analysis=False):
    """Snapshot results from the repo; returns the local root directory.

    sweeps_only=True (default) pulls just the per-pair JSONs (sweep_results,
    procrustes_sweep) plus the root-level consolidated JSONs — under 100 MB.
    False pulls everything, including the figures.

    With sweeps_only=True, two opt-in additions:
      include_dumps     the three rebuttal activation dumps
                        (sweep-results/activation_dumps/*.npz, ~280 MB)
      include_analysis  the analysis CSVs written by the experiment notebook
                        (sweep-results/analysis_*/*.csv: Tables A-F), which
                        are the regression targets for the rebuttal numbers


    """
    if pid:
        patterns = [f"{RESULTS_PREFIX}/{pid}/*"]
    elif sweeps_only:
        patterns = [f"{RESULTS_PREFIX}/*/*.json", "*.json"]
        if include_dumps:
            patterns.append(f"{RESULTS_PREFIX}/{DUMPS_DIR}/*.npz")
        if include_analysis:
            patterns.append(f"{RESULTS_PREFIX}/analysis_*/*.csv")
    else:
        patterns = None
    return snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        allow_patterns=patterns,
        token=token or os.environ.get("HF_TOKEN"),
        local_dir=local_dir,
    )
 
 
def upload_pair(root, pid, repo_id=RESULTS_REPO, token=None, private=True,
                extra_files=()):
    """Push one pair's artifacts to the results repo in a single commit.
 
    One commit per pair rather than one per file: the original did seven
    separate upload_file calls, which is seven commits and seven chances to
    leave a pair half-written.
    """
    from huggingface_hub.hf_api import CommitOperationAdd
 
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", private=private,
                exist_ok=True, token=token)
 
    d = pair_dir(root, pid)
    ops = [
        CommitOperationAdd(path_in_repo=repo_path(pid, f), path_or_fileobj=str(d / f))
        for f in list(EXPECTED_FILES) + list(extra_files)
        if (d / f).exists()
    ]
    if not ops:
        print(f"upload_pair: nothing to upload for {pid}")
        return
 
    api.create_commit(repo_id, repo_type="dataset", operations=ops,
                      commit_message=f"results: {pid}", token=token)
    print(f"uploaded {len(ops)} files for {pid}")


# --- adapter revisions ------------------------------------------------------

class AdapterMismatchError(RuntimeError):
    """The adapter repo serves different weights than the ones a saved run used."""


def resolve_revision(repo_id, revision=None, token=None):
    """Branch, tag, or None (HEAD) -> the commit hash the Hub serves for it now."""
    return HfApi(token=token or os.environ.get("HF_TOKEN")).model_info(repo_id, revision=revision).sha


def verify_adapter(payload, revision=None, token=None):
    """Confirm the adapter a saved run used is what the repo serves; return its commit hash.

    Looks up the adapter at `revision`, else at the revision the run recorded,
    else HEAD, downloads only its config.json, and compares it with the
    `adapter.config` saved in the payload (the config.json loaded at run time,
    including the training metrics). A retrained checkpoint differs there even
    with identical hyperparameters. Raises AdapterMismatchError on a mismatch.
    """
    from huggingface_hub import hf_hub_download

    ad = payload.get("adapter") or {}
    repo, sub = ad.get("repo"), ad.get("subfolder")
    if not repo or not sub:
        raise ValueError(f"{payload.get('pair_id')}: no adapter repo/subfolder recorded")
    requested = revision or ad.get("revision")
    tok = token or os.environ.get("HF_TOKEN")
    sha = resolve_revision(repo, requested, token=tok)
    recorded = ad.get("config")
    if recorded is None:
        return sha
    with open(hf_hub_download(repo_id=repo, filename="config.json", subfolder=sub,
                              revision=sha, token=tok)) as f:
        current = json.load(f)
    if current != recorded:
        keys = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
        fvu = lambda c: (c.get("metrics") or {}).get("fvu")
        raise AdapterMismatchError(
            f"{payload.get('pair_id')}: {repo}@{sha[:10]} serves a different checkpoint than "
            f"this run used (differs in {keys}; fvu recorded {fvu(recorded)} vs now {fvu(current)}). "
            + ("The run recorded no revision, so HEAD was checked; if the repo has history, pass "
               "revision= the commit whose config matches." if not requested else
               f"Requested revision {requested!r}."))
    return sha