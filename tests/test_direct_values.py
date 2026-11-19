import copy
import numpy as np
import pytest
from safetensors.numpy import save_file
from weight_space_lab import core


def plan(root):
    w = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    save_file({"w": w}, str(root / "source.safetensors"))
    return w, {"checkpoint": "source.safetensors", "source_sha256": core.sha256(root / "source.safetensors"),
        "hypothesis": "Explicit LLM-written coordinates, no gradient or automatic fitting",
        "limits": {"max_changed_fraction": .5, "max_relative_l2": .5},
        "operations": [{"kind": "set_values", "tensor": "w", "rows": [2, 0], "columns": [3, 1],
            "expected_values": [[12, 10], [4, 2]], "values": [[11, 9], [3, 1]]}]}


def test_explicit_values_and_order_and_identity(tmp_path):
    original, p = plan(tmp_path)
    actual, report = core.prepare(tmp_path, p)
    np.testing.assert_array_equal(actual["w"], [[1, 1, 3, 3], [5, 6, 7, 8], [9, 9, 11, 11]])
    assert report["changed_parameters"] == 4
    p["operations"][0]["values"] = p["operations"][0]["expected_values"]
    actual, report = core.prepare(tmp_path, p)
    np.testing.assert_array_equal(actual["w"], original)
    assert report["changed_parameters"] == 0


@pytest.mark.parametrize("field,value", [("expected_values", [[0, 0], [0, 0]]),
    ("values", [[float('nan'), 0], [0, 0]]), ("values", [[1, 2]]),
    ("rows", [True, 0]), ("columns", [1, 1])])
def test_explicit_value_rejections(tmp_path, field, value):
    _, p = plan(tmp_path)
    p["operations"][0][field] = value
    with pytest.raises(ValueError):
        core.prepare(tmp_path, p)


def test_values_still_obey_budgets(tmp_path):
    _, p = plan(tmp_path)
    for key in ("max_changed_fraction", "max_relative_l2"):
        q = copy.deepcopy(p); q["limits"][key] = .001
        with pytest.raises(ValueError, match="budget"):
            core.write_candidate(tmp_path, q, "rejected")
    assert not (tmp_path / "rejected").exists()
