import json
import numpy as np
import pytest
from test_diagnostics import tiny_world
from weight_space_lab import core, evaluation as ev
from weight_space_lab.inference import describe_parameter, probe_model, score_candidate


def test_explicit_edit_forward_and_gpu_without_training(tmp_path):
    tiny_world(tmp_path)
    tensor = "transformer.h.0.mlp.c_proj.weight"
    read = describe_parameter(tmp_path, "model", tensor, [0, 1], [0, 1])
    assert read["layout"] == "in,out" and read["input_feature_axis"] == 0
    values = (np.asarray(read["values"]) + .001).tolist()
    plan = {"checkpoint": "model/model.safetensors", "source_sha256": read["source_sha256"],
        "hypothesis": "Directly adjust inspected projection coordinates, measured through inference only",
        "limits": {"max_changed_fraction": .1, "max_relative_l2": .1},
        "operations": [{"kind": "set_values", "tensor": tensor, "rows": [0, 1], "columns": [0, 1],
            "expected_values": read["values"], "values": values}]}
    core.write_candidate(tmp_path, plan, "candidate")
    read_after = describe_parameter(tmp_path, "model", tensor, [0, 1], [0, 1], "candidate")
    np.testing.assert_allclose(read_after["values"], values, atol=1e-8)
    model, _ = ev.apply_candidate(tmp_path, "model", "candidate")
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    observed = []
    import torch
    original = model.forward
    def forward(*a, **kw):
        observed.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return original(*a, **kw)
    model.forward = forward
    ev.evaluate_model(model, ev.load_dataset(tmp_path, "data.json"), ["discovery"])
    assert observed and all(not enabled and inference for enabled, inference in observed)
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        result = probe_model(tmp_path, "model", "data.json", "candidate", max_new_tokens=2, device=device)
        assert result["device"] == device and len(result["examples"][0]["continuation_ids"]) == 2
        score = score_candidate(tmp_path, "model", "candidate", "data.json", ["validation"], device=device)
        assert score["optimizer_steps"] == 0 and not score["gradients_enabled"]
    assert core.sha256(tmp_path / "model/model.safetensors") == read["source_sha256"]
    with pytest.raises(ValueError, match="discovery/validation"):
        score_candidate(tmp_path, "model", "candidate", "data.json", ["test"])
