import json

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from weight_space_lab.core import inspect, prepare, read_matrix, sha256, write_candidate


@pytest.fixture
def experiment(tmp_path):
    weights = np.array([[1, 2], [3, 4]], dtype=np.float32)
    save_file({"mlp.weight": weights}, str(tmp_path / "model.safetensors"))
    plan = {"checkpoint": "model.safetensors", "source_sha256": sha256(tmp_path / "model.safetensors"),
            "hypothesis": "test a localized output direction", "limits": {
                "max_changed_fraction": 0.5, "max_relative_l2": 0.5},
            "operations": [{"tensor": "mlp.weight", "kind": "project_out", "axis": 1,
                            "indices": [0], "direction": [1, 0], "strength": 1.0}]}
    return tmp_path, weights, plan


def test_projection_locality_and_provenance(experiment):
    root, weights, plan = experiment
    before = (root / "model.safetensors").read_bytes()
    report = write_candidate(root, plan, "runs/candidate")
    actual = load_file(str(root / "runs/candidate/replacements.safetensors"))["mlp.weight"]
    np.testing.assert_array_equal(actual, [[0, 2], [3, 4]])
    assert report["changed_parameters"] == 1
    assert report["status"] == "unevaluated"
    assert (root / "model.safetensors").read_bytes() == before
    manifest = json.loads((root / "runs/candidate/manifest.json").read_text())
    assert manifest["report"]["patch_sha256"] == sha256(root / "runs/candidate/replacements.safetensors")
    with pytest.raises(ValueError, match="already exists"):
        write_candidate(root, plan, "runs/candidate")


def test_read_and_inspect(experiment):
    root, _, plan = experiment
    assert inspect(root, plan["checkpoint"])["tensors"][0]["shape"] == [2, 2]
    assert read_matrix(root, plan["checkpoint"], "mlp.weight", [1], [0])["values"] == [[3.0]]
    with pytest.raises(ValueError):
        read_matrix(root, plan["checkpoint"], "mlp.weight", [-1], [0])


@pytest.mark.parametrize("change,reason", [
    ({"source_sha256": "stale"}, "hash mismatch"),
    ({"limits": {"max_changed_fraction": .01, "max_relative_l2": .5}}, "parameter budget"),
    ({"limits": {"max_changed_fraction": .5, "max_relative_l2": .01}}, "L2 budget"),
    ({"limits": {"max_changed_fraction": float('nan'), "max_relative_l2": .5}}, "finite"),
])
def test_rejects_stale_or_over_budget(experiment, change, reason):
    root, _, plan = experiment
    plan.update(change)
    with pytest.raises(ValueError, match=reason):
        write_candidate(root, plan, "rejected")
    assert not (root / "rejected").exists()


def test_identity_and_scale(experiment):
    root, weights, plan = experiment
    operation = plan["operations"][0]
    operation.update(kind="scale", axis=0, strength=0)
    candidate, report = prepare(root, plan)
    np.testing.assert_array_equal(candidate["mlp.weight"], weights)
    assert report["changed_parameters"] == 0
    operation["strength"] = .5
    candidate, _ = prepare(root, plan)
    np.testing.assert_array_equal(candidate["mlp.weight"], [[.5, 1], [3, 4]])


def test_invalid_direction_and_paths(experiment):
    root, _, plan = experiment
    for direction in ([0, 0], [1], [float('nan'), 0]):
        plan["operations"][0]["direction"] = direction
        with pytest.raises(ValueError, match="direction"):
            prepare(root, plan)
    with pytest.raises(ValueError, match="inside workspace"):
        inspect(root, "../model.safetensors")
    outside = root.parent / "outside"
    outside.mkdir(exist_ok=True)
    (root / "link").symlink_to(outside, target_is_directory=True)
    plan["operations"][0]["direction"] = [1, 0]
    with pytest.raises(ValueError, match="inside workspace"):
        write_candidate(root, plan, "link/candidate")


def test_row_projection(experiment):
    root, _, plan = experiment
    plan["operations"][0].update(axis=0, indices=[1], direction=[0, 1], strength=.5)
    candidate, _ = prepare(root, plan)
    np.testing.assert_array_equal(candidate["mlp.weight"], [[1, 2], [3, 2]])


def test_readonly_hash_cache_preserves_mutation_detection(tmp_path,monkeypatch):
    import hashlib,os
    from pathlib import Path
    from weight_space_lab import core
    p=tmp_path/'immutable';p.write_bytes(b'original');p.chmod(0o444)
    first=core.sha256(p);assert first==hashlib.sha256(b'original').hexdigest()
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',lambda *a,**k:pytest.fail('immutable second read must avoid content I/O'))
        assert core.sha256(p)==first
    before=p.stat();p.chmod(0o644);p.write_bytes(b'modified');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns));p.chmod(0o444)
    assert core.sha256(p)==hashlib.sha256(b'modified').hexdigest()!=first
