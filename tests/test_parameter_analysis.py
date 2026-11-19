import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from safetensors.numpy import save_file
from transformers.pytorch_utils import Conv1D

from weight_space_lab import analysis, core


def world(root, monkeypatch, conv=False):
    class KnownLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=8, n_positions=8)
            self.embedding = torch.nn.Embedding(8, 2)
            self.projection = Conv1D(2, 2) if conv else torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 8, bias=False)
            with torch.no_grad():
                self.embedding.weight.fill_(1)
                self.embedding.weight[1] = torch.tensor([1., -1.])
                self.embedding.weight[2] = torch.tensor([-2., 3.])
                w = torch.tensor([[.4, .2], [-.1, .3]])
                self.projection.weight.copy_(w.T if conv else w)
                self.projection.bias.copy_(torch.tensor([.5, -.25]))
                self.head.weight.fill_(.1)
            self.eval().requires_grad_(False)

        def forward(self, input_ids):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            return SimpleNamespace(logits=self.head(self.projection(self.embedding(input_ids))))

    model = KnownLM()
    (root / "model").mkdir()
    (root / "model/config.json").write_text("{}")
    save_file({k: v.numpy() for k, v in model.state_dict().items()}, str(root / "model/model.safetensors"))
    data = {"splits": {"discovery": [{"id": "reference", "tokens": [1, 3, 4]},
                                      {"id": "failure", "tokens": [2, 3, 4]}],
                       "test": [{"id": "sealed", "tokens": [5, 3, 4]}]}}
    (root / "data.json").write_text(json.dumps(data))
    monkeypatch.setattr(analysis, "load_inference_model", lambda *a, **k: model)
    return model


def spec():
    return {"model_dir": "model", "dataset_path": "data.json", "tensor": "projection.weight",
        "rows": [0], "columns": [0], "pairs": [{"id": "contrast", "reference_id": "reference",
            "failure_id": "failure", "reference_positions": [0], "failure_positions": [0]}],
        "proposed_values": [[.2]]}


def basis():
    return {"failure_description": "Declared failure differs at first token; correctness is not inferred.",
        "structural_role": "Affine projection from selected input feature to selected output feature.",
        "mechanism_hypothesis": "Signed input reverses the selected feature's residual contribution.",
        "pairing_justification": "Controlled two-feature test fixture; first input token differs.",
        "value_derivation": "LLM-selected 50% attenuation, .4 to .2; test signed output effect.",
        "alternative_explanation": "Other features/bias or downstream behavior may dominate.",
        "expected_behavior": "Positive local shift in failure, bounded reference change.",
        "disconfirming_result": "Actual forward misses the local prediction, or behavioral metric worsens.",
        "controls": [{"kind": "identity", "purpose": "unchanged baseline"},
                     {"kind": "opposing_edit", "purpose": "reverse predicted local effect"}],
        "evidence_pairs": ["contrast"], "value_rule": {"kind": "attenuate", "fraction": .5},
        "target_output_direction": [1.], "max_reference_output_delta": .21}


@pytest.mark.parametrize("conv", [False, True])
def test_signed_position_contributions_and_probe_proposal(tmp_path, monkeypatch, conv):
    model = world(tmp_path, monkeypatch, conv)
    before = core.sha256(tmp_path / "model/model.safetensors")
    response = analysis.analyze_parameter_contexts(tmp_path, **spec(), output="analysis.json")
    report = response["report"]
    pair = report["observations"][0]
    assert report["layout"] == ("in,out" if conv else "out,in")
    assert pair["reference"]["signed_input"] == [1.]
    assert pair["failure"]["signed_input"] == [-2.]
    np.testing.assert_allclose(pair["reference"]["selected_weight_contribution"], [.4], atol=1e-7)
    np.testing.assert_allclose(pair["failure"]["selected_weight_contribution"], [-.8], atol=1e-7)
    np.testing.assert_allclose(pair["reference"]["fixed_input_output_delta"], [-.2], atol=1e-7)
    np.testing.assert_allclose(pair["failure"]["fixed_input_output_delta"], [.4], atol=1e-7)
    # Independent full affine evaluation at the actual observed inputs; no derivatives.
    w = model.projection.weight.detach().clone()
    w[0, 0] = .2
    with torch.inference_mode():
        for token, role in [(1, "reference"), (2, "failure")]:
            x = model.embedding(torch.tensor([token]))
            old_y = model.projection(x)
            new_y = x @ w + model.projection.bias if conv else x @ w.T + model.projection.bias
            np.testing.assert_allclose((new_y - old_y)[0, :1], pair[role]["fixed_input_output_delta"], atol=1e-7)
    card = analysis.prepare_edit_proposal(tmp_path, "analysis.json", basis(),
        {"max_changed_fraction": 1, "max_relative_l2": 1}, "proposal.json")
    assert card["decision"] == "ready_for_causal_probe"
    assert card["causal_status"] == card["performance_status"] == "unverified"
    assert card["plan"]["operations"][0]["values"] == [[float(np.float32(.2))]]
    assert not card["candidate_created"]
    assert core.sha256(tmp_path / "model/model.safetensors") == before
    assert not model.projection._forward_hooks


@pytest.mark.parametrize("change", ["missing_reason", "wrong_direction", "reference_regression",
                                  "false_arithmetic", "missing_control", "unknown_evidence"])
def test_defer_incomplete_or_conflicting_basis(tmp_path, monkeypatch, change):
    world(tmp_path, monkeypatch)
    analysis.analyze_parameter_contexts(tmp_path, **spec(), output="a.json")
    b = basis()
    if change == "missing_reason": b.pop("value_derivation")
    if change == "wrong_direction": b["target_output_direction"] = [-1.]
    if change == "reference_regression": b["max_reference_output_delta"] = .01
    if change == "false_arithmetic": b["value_rule"]["fraction"] = .1
    if change == "missing_control": b["controls"] = []
    if change == "unknown_evidence": b["evidence_pairs"] = ["invented"]
    card = analysis.prepare_edit_proposal(tmp_path, "a.json", b,
        {"max_changed_fraction": 1, "max_relative_l2": 1}, "defer.json")
    assert card["decision"] == "defer" and card["reasons"]
    assert card["plan"] is None and not card["candidate_created"]


def test_explicit_coordinate_reason_and_stale_evidence(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    analysis.analyze_parameter_contexts(tmp_path, **spec(), output="a.json")
    b = basis()
    b["value_rule"] = {"kind": "explicit", "coordinate_reasons": [
        {"row": 0, "column": 0, "reason": "Reduce the recorded signed component by .2."}]}
    card = analysis.prepare_edit_proposal(tmp_path, "a.json", b,
        {"max_changed_fraction": 1, "max_relative_l2": 1}, "explicit.json")
    assert card["decision"] == "ready_for_causal_probe"
    (tmp_path / "data.json").write_text((tmp_path / "data.json").read_text() + " ")
    with pytest.raises(ValueError, match="stale"):
        analysis.prepare_edit_proposal(tmp_path, "a.json", b, {}, "stale.json")
    assert not (tmp_path / "stale.json").exists()


def test_rank_one_joint_reason_is_checked_without_fitting(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    analysis.analyze_parameter_contexts(tmp_path, **spec(), output="a.json")
    b = basis()
    b["value_rule"] = {"kind": "rank_one", "left": [-1.], "right": [.2], "strength": 1.}
    card = analysis.prepare_edit_proposal(tmp_path, "a.json", b,
        {"max_changed_fraction": 1, "max_relative_l2": 1}, "rank.json")
    assert card["decision"] == "ready_for_causal_probe"
    b["value_rule"]["right"] = [.1]
    card = analysis.prepare_edit_proposal(tmp_path, "a.json", b,
        {"max_changed_fraction": 1, "max_relative_l2": 1}, "wrong-rank.json")
    assert card["decision"] == "defer" and card["plan"] is None


def test_identical_contexts_defer_and_no_proposal_defer(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    s = spec(); s["pairs"][0]["failure_id"] = "reference"
    analysis.analyze_parameter_contexts(tmp_path, **s, output="same.json")
    card = analysis.prepare_edit_proposal(tmp_path, "same.json", basis(),
        {"max_changed_fraction": 1, "max_relative_l2": 1}, "same-basis.json")
    assert card["decision"] == "defer" and not card["local_checks"]["nonidentical_contexts"]
    s = spec(); s["proposed_values"] = None
    analysis.analyze_parameter_contexts(tmp_path, **s, output="observe.json")
    card = analysis.prepare_edit_proposal(tmp_path, "observe.json", basis(), {}, "observe-basis.json")
    assert card["decision"] == "defer" and card["plan"] is None


@pytest.mark.parametrize("bad", ["heldout", "bad_position", "mismatched_counts", "nan"])
def test_analysis_rejects_bad_inputs(tmp_path, monkeypatch, bad):
    world(tmp_path, monkeypatch)
    s = spec()
    if bad == "heldout": s["pairs"][0]["failure_id"] = "sealed"
    if bad == "bad_position": s["pairs"][0]["reference_positions"] = [2]
    if bad == "mismatched_counts": s["pairs"][0]["failure_positions"] = [0, 1]
    if bad == "nan": s["proposed_values"] = [[float('nan')]]
    with pytest.raises(ValueError):
        analysis.analyze_parameter_contexts(tmp_path, **s, output="bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_real_transformer_intervention_matches_local_affine_prediction(tmp_path):
    from test_diagnostics import tiny_world
    tiny_world(tmp_path)
    name = "transformer.h.0.mlp.c_proj.weight"
    old = core.read_matrix(tmp_path, "model/model.safetensors", name, [0], [0])["values"]
    new = [[old[0][0] + .005]]
    s = {"model_dir": "model", "dataset_path": "data.json", "tensor": name,
         "rows": [0], "columns": [0], "pairs": [{"id": "numerical-control", "reference_id": "d",
            "failure_id": "d", "reference_positions": [0], "failure_positions": [0]}]}
    before = analysis.analyze_parameter_contexts(tmp_path, **s, proposed_values=new, output="before.json")["report"]
    plan = {"checkpoint": "model/model.safetensors", "source_sha256": before["source_sha256"],
        "hypothesis": "Exact affine-intervention numerical control, not semantic learning evidence",
        "limits": {"max_changed_fraction": .1, "max_relative_l2": .1},
        "operations": [{"kind": "set_values", "tensor": name, "rows": [0], "columns": [0],
                        "expected_values": old, "values": new}]}
    core.write_candidate(tmp_path, plan, "probe")
    after = analysis.analyze_parameter_contexts(tmp_path, **s, candidate_dir="probe", output="after.json")["report"]
    b, a = before["observations"][0]["reference"], after["observations"][0]["reference"]
    np.testing.assert_allclose(np.asarray(a["actual_module_output"]) - b["actual_module_output"],
                               b["fixed_input_output_delta"], atol=1e-7)
    assert after["patch_sha256"] == core.sha256(tmp_path / "probe/replacements.safetensors")
    assert a["target_log_probability"] != b["target_log_probability"]
    with pytest.raises(ValueError, match="original analysis"):
        analysis.prepare_edit_proposal(tmp_path, "after.json", {}, {}, "invalid-proposal.json")


def test_failed_forward_cleans_up_hooks_and_does_not_write_report(tmp_path, monkeypatch):
    model = world(tmp_path, monkeypatch)
    def failed_forward(*args, **kwargs):
        raise RuntimeError("forward failed")
    model.forward = failed_forward
    with pytest.raises(RuntimeError, match="forward failed"):
        analysis.analyze_parameter_contexts(tmp_path, **spec(), output="failed.json")
    assert not model.projection._forward_hooks
    assert not (tmp_path / "failed.json").exists()


def test_context_evidence_deduplicates_relabelled_positions(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    data = json.loads((tmp_path / "data.json").read_text())
    data["splits"]["discovery"] += [{"id": "ref-alias", "tokens": [1, 3, 4]}, {"id": "fail-alias", "tokens": [2, 3, 4]}]
    (tmp_path / "data.json").write_text(json.dumps(data))
    args = spec()
    first = args['pairs'][0]
    args['pairs'] = [{**first, 'reference_positions': [0, 1], 'failure_positions': [0, 1]},
                     {**first, 'id': 'relabeled', 'reference_id': 'ref-alias', 'failure_id': 'fail-alias', 'reference_positions': [0, 1], 'failure_positions': [0, 1]}]
    report = analysis.analyze_parameter_contexts(tmp_path, **args, output='grouped.json')['report']
    evidence = report['context_evidence']
    assert evidence['observation_rows'] == 4
    assert evidence['distinct_sequence_pairs'] == 1
    assert evidence['unique_position_pairs'] == 2
    assert evidence['contexts'][0]['mean_reference_signed_input'] == [1.]
    assert evidence['contexts'][0]['mean_failure_signed_input'] == [-.5]
    np.testing.assert_allclose(evidence['contexts'][0]['mean_contribution_contrast'], [-.6], atol=1e-7)
    assert 'not established' in evidence['independence_status']
