import json
from types import SimpleNamespace
import numpy as np
import pytest
from safetensors.numpy import save_file
from weight_space_lab import computation, core


def known_world(root, monkeypatch, width=2):
    torch = pytest.importorskip("torch")

    class ResidualLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=8, n_positions=8, model_type="known-fixture")
            self.embed = torch.nn.Embedding(8, width)
            self.proj = torch.nn.Linear(width, width)
            self.head = torch.nn.Linear(width, 8, bias=False)
            with torch.no_grad():
                self.embed.weight.fill_(1.)
                self.embed.weight[1, :2] = torch.tensor([2., -1.])
                self.proj.weight.zero_()
                self.proj.weight[:2, :2].copy_(torch.tensor([[.4, .2], [-.1, .3]]))
                self.proj.bias.zero_()
                self.proj.bias[:2].copy_(torch.tensor([.5, -.25]))
                self.head.weight.fill_(.1)
            self.eval().requires_grad_(False)

        def forward(self, input_ids):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            x = self.embed(input_ids)
            return SimpleNamespace(logits=self.head(x + self.proj(x)))

    model = ResidualLM()
    (root / "model").mkdir()
    (root / "model/config.json").write_text("{}")
    save_file({k: v.numpy() for k, v in model.state_dict().items()}, root / "model/model.safetensors")
    (root / "data.json").write_text(json.dumps({"splits": {
        "discovery": [{"id": "d", "tokens": [1, 2, 3]}],
        "test": [{"id": "sealed", "tokens": [1, 3, 4]}]}}))
    monkeypatch.setattr(computation, "load_inference_model", lambda *a, **k: model)
    return model


def test_actual_source_vector_reads_and_hierarchy(tmp_path, monkeypatch):
    known_world(tmp_path, monkeypatch)
    values = computation.read_parameter_entries(tmp_path, "model/model.safetensors", "proj.bias", [1, 0])
    assert values["entries"] == [{"flat_index": 1, "coordinates": [1], "value": -.25},
                                {"flat_index": 0, "coordinates": [0], "value": .5}]
    structure = computation.describe_model_structure(tmp_path, "model", max_modules=2)
    assert structure["next_offset"] == 2 and len(structure["modules"]) == 2
    source = computation.inspect_module_computation(tmp_path, "model", ["", "proj"])
    assert "x + self.proj(x)" in source["modules"][0]["forward_source"]
    assert source["modules"][1]["equation"] == "y = x @ W.T + bias; W[out,in]"
    assert source["model_forward_calls"] == 0


def test_signed_forward_values_and_no_invented_residual_edge(tmp_path, monkeypatch):
    model = known_world(tmp_path, monkeypatch)
    specs = [{"module": n, "input_features": [0, 1], "output_features": [0, 1]}
             for n in ["embed", "proj", "head"]]
    result = computation.trace_token_flow(tmp_path, "model", "data.json", specs,
        [{"id": "d", "positions": [0, 1]}], "flow.json")["report"]
    events = result["examples"][0]["events"]
    assert [e["module"] for e in events] == ["embed", "proj", "head"]
    np.testing.assert_allclose(events[1]["input"]["samples"][0]["signed_values"], [2., -1.])
    np.testing.assert_allclose(events[1]["output"]["samples"][0]["signed_values"], [1.1, -.75], atol=1e-7)
    assert events[1]["input_same_tensor_as_event_output"] == 0
    # head receives x + proj(x), a new tensor from an unhooked functional operation.
    assert events[2]["input_same_tensor_as_event_output"] is None
    assert events[2]["input_link_status"] == "unobserved_or_functional_path"
    assert not result["gradients_enabled"] and result["optimizer_steps"] == 0
    assert not model.proj._forward_hooks and not model.proj._forward_pre_hooks
    assert not (tmp_path / "flow.json").is_symlink()


def test_flow_rejects_holdout_and_cleans_hooks_on_bad_features(tmp_path, monkeypatch):
    model = known_world(tmp_path, monkeypatch)
    specs = [{"module": "proj", "input_features": [99], "output_features": [0]}]
    with pytest.raises(ValueError, match="development"):
        computation.trace_token_flow(tmp_path, "model", "data.json", specs, [], "bad.json", split="test")
    with pytest.raises(ValueError, match="indices"):
        computation.trace_token_flow(tmp_path, "model", "data.json", specs,
            [{"id": "d", "positions": [0]}], "bad.json")
    assert not model.proj._forward_hooks and not model.proj._forward_pre_hooks
    assert not (tmp_path / "bad.json").exists()


def test_weight_geometry_reconstructs_block_and_composes_without_forward(tmp_path):
    a = np.array([[1., 2.], [-3., 4.]], dtype=np.float32)
    b = np.array([[2., 0.], [0., 3.]], dtype=np.float32)
    save_file({"a": a, "b": b}, tmp_path / "w.safetensors")
    result = computation.analyze_matrix_geometry(tmp_path, "w.safetensors", "a", [0, 1], [0, 1], rank=2,
        right_operand={"tensor": "b", "rows": [0, 1], "columns": [0, 1]})
    reconstructed = np.asarray(result["left_vectors"]) @ np.diag(result["singular_values"]) @ result["right_vectors"]
    np.testing.assert_allclose(reconstructed, a, atol=1e-12)
    np.testing.assert_array_equal(result["product"], a @ b)
    assert result["model_forward_calls"] == result["parameter_updates"] == 0


def test_sparse_entries_support_higher_rank_and_scalar(tmp_path):
    save_file({"cube": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
               "scalar": np.array(7., dtype=np.float32)}, tmp_path / "t.safetensors")
    result = computation.read_parameter_entries(tmp_path, "t.safetensors", "cube", [0, 17, 23])
    assert [e["coordinates"] for e in result["entries"]] == [[0, 0, 0], [1, 1, 1], [1, 2, 3]]
    assert [e["value"] for e in result["entries"]] == [0., 17., 23.]
    scalar = computation.read_parameter_entries(tmp_path, "t.safetensors", "scalar", [0])
    assert scalar["shape"] == [] and scalar["entries"][0]["coordinates"] == []
    assert scalar["entries"][0]["value"] == 7.


def test_llm_coefficient_can_exceed_one_without_escaping_budget(tmp_path):
    a = np.array([[.01, 1.], [.02, 2.]], dtype=np.float32)
    save_file({"a": a}, tmp_path / "w.safetensors")
    plan = {"checkpoint": "w.safetensors", "source_sha256": core.sha256(tmp_path / "w.safetensors"),
        "hypothesis": "explicit overprojection test, no training", "limits": {
            "max_changed_fraction": .5, "max_relative_l2": .1}, "operations": [{
            "kind": "project_out", "tensor": "a", "axis": 1, "indices": [0],
            "direction": [1., 0.], "strength": 6.}]}
    values, report = core.prepare(tmp_path, plan)
    assert values["a"][0, 0] < 0
    np.testing.assert_allclose(values["a"][0, 0], -.05, atol=1e-8)
    plan["limits"]["max_relative_l2"] = .001
    with pytest.raises(ValueError, match="budget"):
        core.prepare(tmp_path, plan)


def test_wide_flow_and_total_budget(tmp_path, monkeypatch):
    known_world(tmp_path, monkeypatch, width=80)
    specs = [{"module": "proj", "input_features": list(range(80)), "output_features": [0]}]
    flow = computation.trace_token_flow(tmp_path, "model", "data.json", specs,
        [{"id": "d", "positions": [0]}], "wide.json")["report"]
    assert len(flow["examples"][0]["events"][0]["input"]["samples"][0]["signed_values"]) == 80
    data = {"splits": {"discovery": [{"id": str(i), "tokens": list(range(10))} for i in range(4)]}}
    (tmp_path / "wide-data.json").write_text(json.dumps(data))
    monkeypatch.setattr(computation, "load_inference_model", lambda *a, **k: pytest.fail("must reject before model load"))
    with pytest.raises(ValueError, match="scalar budget"):
        computation.trace_token_flow(tmp_path, "model", "wide-data.json",
            [{"module": str(i), "input_features": list(range(512)), "output_features": list(range(512))} for i in range(3)],
            [{"id": str(i), "positions": list(range(8))} for i in range(4)], "bad.json")
