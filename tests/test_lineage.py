import numpy as np
from safetensors import safe_open
from test_diagnostics import tiny_world
from weight_space_lab import core
from weight_space_lab.lineage import compare_candidate_parameters
from weight_space_lab.lineage import revise_candidate


def test_retention_repair_rollback_and_new_changes_are_distinct(tmp_path):
    tiny_world(tmp_path)
    tensor = "transformer.h.0.mlp.c_proj.weight"
    checkpoint = "model/model.safetensors"
    with safe_open(tmp_path / checkpoint, framework="np") as f:
        original = f.get_tensor(tensor)[:1, :4].copy()
    parent = original.copy()
    parent[0, :3] += np.float32(.001)
    child = parent.copy()
    child[0, 1] += np.float32(.001)  # revised previous edit
    child[0, 2] = original[0, 2]    # restored
    child[0, 3] += np.float32(.001)  # new
    digest = core.sha256(tmp_path / checkpoint)
    for name, values in (("parent", parent), ("child", child)):
        core.write_candidate(tmp_path, {"checkpoint": checkpoint, "source_sha256": digest,
            "hypothesis": "declared fixture values, no training", "limits": {
                "max_changed_fraction": .1, "max_relative_l2": .1}, "operations": [{
                "kind": "set_values", "tensor": tensor, "rows": [0], "columns": [0, 1, 2, 3],
                "expected_values": original.tolist(), "values": values.tolist()}]}, name)
    result = compare_candidate_parameters(tmp_path, "model", "child", "parent")
    assert result["incremental_changed"] == 3
    assert result["net_changed"] == 3
    assert result["retained_parent_changes"] == result["revised_parent_changes"] == 1
    assert result["restored_to_original"] == result["new_changes"] == 1
    assert core.sha256(tmp_path / checkpoint) == digest
    noop = compare_candidate_parameters(tmp_path, "model", "parent", "parent")
    assert noop["incremental_changed"] == 0 and noop["retained_parent_changes"] == 3


def test_incremental_revision_preserves_parent_and_checks_current_values(tmp_path):
    import json
    import pytest
    tiny_world(tmp_path)
    tensor, checkpoint = "transformer.h.0.mlp.c_proj.weight", "model/model.safetensors"
    with safe_open(tmp_path / checkpoint, framework="np") as f:
        original = f.get_tensor(tensor)[:1, :2]
    parent = (original + np.float32(.001)).astype(np.float32)
    digest = core.sha256(tmp_path / checkpoint)
    limits = {"max_changed_fraction": .1, "max_relative_l2": .1}
    core.write_candidate(tmp_path, {"checkpoint": checkpoint, "source_sha256": digest,
        "hypothesis": "explicit fixture edit", "limits": limits, "operations": [{
        "tensor": tensor, "kind": "set_values", "rows": [0], "columns": [0, 1],
        "expected_values": original.tolist(), "values": parent.tolist()}]}, "parent")
    operation = {"tensor": tensor, "kind": "set_values", "rows": [0], "columns": [0],
                 "expected_values": [[float(parent[0, 0])]], "values": [[float(parent[0, 0] + .001)]]}
    rationale = {k: "known fixture arithmetic, no behavioral or learning claim" for k in (
        "observation", "hypothesis", "value_derivation", "prediction", "disconfirming_result")}
    revise_candidate(tmp_path, "model", "parent", [operation], "explicit child", rationale, limits, "child")
    traced = compare_candidate_parameters(tmp_path, "model", "child", "parent")
    assert traced["retained_parent_changes"] == traced["revised_parent_changes"] == 1
    assert traced["incremental_changed"] == 1 and traced["net_changed"] == 2
    manifest = json.loads((tmp_path / "child/manifest.json").read_text())
    assert manifest["lineage"]["parent_patch_sha256"] == traced["parent_patch_sha256"]
    assert manifest["plan"]["operations"][0]["kind"] == "sequence"
    operation["expected_values"] = [[float(original[0, 0])]]
    with pytest.raises(ValueError, match="expected_values"):
        revise_candidate(tmp_path, "model", "parent", [operation], "stale", rationale, limits, "bad")
    assert not (tmp_path / "bad").exists() and core.sha256(tmp_path / checkpoint) == digest
