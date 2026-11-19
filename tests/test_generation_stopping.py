from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from weight_space_lab import inference
from test_diagnostics import tiny_world


@pytest.mark.parametrize("stop,expected,reason", [(True, 1, "eos"), (False, 3, "max_new_tokens")])
def test_probe_eos_and_legacy_fixed_length(tmp_path, monkeypatch, stop, expected, reason):
    model = tiny_world(tmp_path)
    model.generation_config.eos_token_id = 2
    def forward(ids):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        logits = torch.zeros(1, ids.shape[1], 16)
        logits[:, :, 2] = 10
        return SimpleNamespace(logits=logits)
    model.forward = forward
    monkeypatch.setattr(inference, "load_inference_model", lambda *a, **k: model)
    report = inference.probe_model(tmp_path, "model", "data.json", max_new_tokens=3, stop_at_eos=stop)
    assert report["examples"][0]["continuation_ids"] == [2] * expected
    assert report["examples"][0]["finish_reason"] == reason
    assert report["examples"][0]["generated_tokens"] == expected


def test_probe_zero_generation_reason(tmp_path):
    tiny_world(tmp_path)
    report = inference.probe_model(tmp_path, "model", "data.json", max_new_tokens=0, stop_at_eos=True)
    assert report["examples"][0]["finish_reason"] == "no_generation"
    assert report["examples"][0]["continuation_ids"] == []


def test_cached_and_uncached_real_model_match(tmp_path):
    tiny_world(tmp_path)
    plain = inference.probe_model(tmp_path, "model", "data.json", max_new_tokens=5)
    cached = inference.probe_model(tmp_path, "model", "data.json", max_new_tokens=5, use_cache=True)
    assert cached["examples"][0]["continuation_ids"] == plain["examples"][0]["continuation_ids"]
    assert cached["use_cache"]
