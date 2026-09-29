"""shared_truth — common code for the Shared Truth experiments.

Lazy imports, no GPU runtime necessary.
"""

import importlib
 
__version__ = "0.1.0"
 
_LAZY = {
    "naming", "probes", "activations", "metrics", "sweep", "plots", "storage", "env",
    "residuals",
}
 
__all__ = sorted(_LAZY)
 
 
def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
 
 
def __dir__():
    return sorted(set(globals()) | _LAZY)
