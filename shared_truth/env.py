"""Runtime pinning for the probe files.

The truth probes are scikit-learn estimators serialized with skops, and they
only behave correctly under the scikit-learn version that saved them. Across
1.8, LogisticRegression dropped its `multi_class` attribute; a probe saved under
1.8 and loaded under 1.6 fails inside predict_proba with
    AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
and the reverse direction can silently change behavior.

No sklearn/skops import at module level, on purpose.
"""

import importlib.metadata
import json
import subprocess
import sys
import zipfile

from huggingface_hub import hf_hub_download

__all__ = ["DEFAULT_PROBE", "recorded_sklearn_version", "probe_sklearn_version", "pin_sklearn"]

# Any probe works; they were all written in one environment. Checked per-probe
# again at load time by probes.load_probe.
DEFAULT_PROBE = "KingTechnician/tiu-acts-llama3.2-3b-instruct-layer-12-probe"


def recorded_sklearn_version(skops_path):
    """scikit-learn version stored inside a .skops file, or None if absent."""
    with zipfile.ZipFile(skops_path) as z:
        schema = json.loads(z.read("schema.json"))
    node = schema.get("content", {}).get("content", {}).get("_sklearn_version")
    if not node:
        return None
    raw = node.get("content")
    return json.loads(raw) if node.get("is_json") else raw


def probe_sklearn_version(repo_id=DEFAULT_PROBE):
    return recorded_sklearn_version(hf_hub_download(repo_id=repo_id, filename="model.skops"))


def _installed(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def pin_sklearn(version=None, with_skops=True):
    """Make the runtime's scikit-learn match the probes. Call before importing sklearn.

    version=None reads it from DEFAULT_PROBE. Returns the pinned version.
    If a different scikit-learn is already imported in this process, installing
    can't take effect until restart, so this raises instead of continuing on the
    wrong version.
    """
    version = version or probe_sklearn_version()
    if version is None:
        raise RuntimeError("probe file records no scikit-learn version; pin manually")

    have = _installed("scikit-learn")
    pkgs = [] if have == version else [f"scikit-learn=={version}"]
    if with_skops and _installed("skops") is None:
        pkgs.append("skops")
    if pkgs:
        print(f"installing {' '.join(pkgs)} (probes were saved with scikit-learn {version}; "
              f"runtime had {have})")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

    if "sklearn" in sys.modules and sys.modules["sklearn"].__version__ != version:
        raise RuntimeError(
            f"scikit-learn {sys.modules['sklearn'].__version__} is already imported in this "
            f"runtime; {version} is now installed but needs a restart. "
            f"Runtime > Restart session, then run from the top."
        )
    print(f"scikit-learn {version} (matches probes)")
    return version