"""CPU tests for the helpers the adapter notebook's regression check relies on."""

import json
import types

import numpy as np
import pytest

from shared_truth import metrics, naming, storage


def _payload(**over):
    p = {"pair_id": "gemma2-2b-instruct_l13_to_qwen2_5-1_5b-instruct_l17__truth",
         "source_model": "google/gemma-2-2b-it", "source_layer": 13,
         "target_model": "Qwen/Qwen2.5-1.5B-Instruct", "target_layer": 17,
         "variant": "truth",
         "adapter": {"repo": "KingTechnician/gemma_2_2b_instruct_l13_to_qwen_2.5_1.5b_instruct_l17",
                     "subfolder": "non_linear-gemma2b_to_qwen5b-13_to_17/model"}}
    p.update(over)
    return p


def test_adapter_from_payload_roundtrip():
    repo, sub, variant, rev = naming.adapter_from_payload(_payload())
    assert sub.endswith("/model") and variant == "truth" and rev is None   # published runs: unpinned
    ad = dict(_payload()["adapter"], revision="abc123")
    assert naming.adapter_from_payload(_payload(adapter=ad))[3] == "abc123"
    # seed-suffixed repos resolve to the same pair
    p = _payload(adapter={"repo": "KingTechnician/gemma_2_2b_instruct_l13_to_qwen_2.5_1.5b_instruct_l17_seed_43",
                          "subfolder": "x/model"}, variant="truth_seed43")
    assert naming.adapter_from_payload(p)[2] == "truth_seed43"


def test_adapter_from_payload_rejects_mismatch_and_baselines():
    with pytest.raises(ValueError, match="resolves to"):
        naming.adapter_from_payload(_payload(target_layer=12))
    with pytest.raises(ValueError, match="no adapter"):
        naming.adapter_from_payload(_payload(adapter={"repo": "r", "note": "random-weight MLP"}))


def _sweep(rng, n1=40, n0=30, preds=True):
    rows = []
    for i in range(11):
        t, f = rng.random(n1), rng.random(n0)
        r = {"alpha": round(i * 0.1, 1), "accuracy": float(rng.random()), "auroc": float(rng.random()),
             "true_scores": t.tolist(), "false_scores": f.tolist()}
        if preds:
            r["preds"] = (np.r_[t, f] > 0.5).astype(int).tolist()
        rows.append(r)
    return rows


def test_sweep_diff_identical_and_changed():
    a = _sweep(np.random.default_rng(0))
    assert all(r["items_differing"] == 0 and r["max_abs_dscore"] == 0 for r in metrics.sweep_diff(a, a))
    b = [dict(r) for r in a]
    b[3] = dict(b[3], preds=list(b[3]["preds"]))
    b[3]["preds"][0] ^= 1
    d = metrics.sweep_diff(b, a)
    assert d[3]["items_differing"] == 1 and d[3]["items_exact"]
    assert sum(r["items_differing"] for r in d) == 1


def test_sweep_diff_old_schema_uses_accuracy():
    a = _sweep(np.random.default_rng(1), preds=False)
    b = [dict(r) for r in a]
    b[5] = dict(b[5], accuracy=b[5]["accuracy"] + 2 / 70)   # two items out of 70
    d = metrics.sweep_diff(b, a)
    assert d[5]["items_differing"] == 2 and not d[5]["items_exact"]


def test_sweep_diff_refuses_different_eval():
    a, b = _sweep(np.random.default_rng(2)), _sweep(np.random.default_rng(3), n1=41)
    with pytest.raises(ValueError, match="item counts differ"):
        metrics.sweep_diff(a, b)


# --- adapter revisions ---------------------------------------------------------

_CFG = {"hidden_dim": 1881, "seed": 42, "source_dim": 2304, "target_dim": 1536,
        "metrics": {"fvu": 0.005051200951001585, "cosine_similarity": 0.7174309265053872}}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Fake Hub: {revision -> (sha, config.json)} for one repo; HEAD is key None."""
    state = {"revs": {None: ("sha_head", _CFG)}, "asked": []}

    class FakeApi:
        def __init__(self, token=None): pass
        def model_info(self, repo_id, revision=None):
            state["asked"].append(revision)
            if revision in state["revs"]:
                return types.SimpleNamespace(sha=state["revs"][revision][0])
            for sha, _ in state["revs"].values():      # a raw commit hash resolves to itself
                if sha == revision:
                    return types.SimpleNamespace(sha=sha)
            raise KeyError(revision)

    def fake_download(repo_id, filename, subfolder=None, revision=None, token=None):
        cfg = next(c for sha, c in state["revs"].values() if sha == revision)
        path = tmp_path / f"{revision}.json"
        path.write_text(json.dumps(cfg))
        return str(path)

    monkeypatch.setattr(storage, "HfApi", FakeApi)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return state


def test_verify_adapter_matching_checkpoint_returns_head_sha(hub):
    p = _payload(adapter=dict(_payload()["adapter"], config=_CFG))
    assert storage.verify_adapter(p) == "sha_head"


def test_verify_adapter_detects_retrained_checkpoint(hub):
    # same hyperparameters, different training outcome -- the floor-pair seed situation
    old = dict(_CFG, metrics={"fvu": 0.0377, "cosine_similarity": 0.689})
    p = _payload(adapter=dict(_payload()["adapter"], config=old))
    with pytest.raises(storage.AdapterMismatchError, match=r"fvu recorded 0.0377 vs now 0.005"):
        storage.verify_adapter(p)


def test_verify_adapter_uses_recorded_revision(hub):
    old = dict(_CFG, metrics={"fvu": 0.0377})
    hub["revs"]["sha_old"] = ("sha_old", old)
    p = _payload(adapter=dict(_payload()["adapter"], config=old, revision="sha_old"))
    assert storage.verify_adapter(p) == "sha_old"                  # pinned run: its own weights
    assert hub["asked"][-1] == "sha_old"
    p_head = _payload(adapter=dict(_payload()["adapter"], config=_CFG, revision="sha_old"))
    with pytest.raises(storage.AdapterMismatchError):
        storage.verify_adapter(p_head)                             # recorded rev holds other weights
    assert storage.verify_adapter(p_head, revision="sha_head") == "sha_head"   # explicit override