import json
import pytest
from test_diagnostics import tiny_world
from test_experiment import config
from weight_space_lab import core
from weight_space_lab.confirmation import confirm_generation
from weight_space_lab.inference import probe_model


def make_frozen_world(root):
    tiny_world(root)
    plan = config(root, 0)["plans"][0]
    core.write_candidate(root, plan, "identity")
    d = {"device": "cpu", "greedy": True, "max_new_tokens": 2, "stop_at_eos": True, "use_cache": True}
    protocol = {"decoding": d, "sealed_files": {"data.json": core.sha256(root / "data.json")}}
    (root / "protocol.json").write_text(json.dumps(protocol))
    frozen = {"kind": "frozen_generation_choice", "model_dir": "model", "selected_candidate": "identity",
        "source_sha256": core.sha256(root / "model/model.safetensors"),
        "config_sha256": core.sha256(root / "model/config.json"),
        "patch_sha256": core.sha256(root / "identity/replacements.safetensors"),
        "protocol_path": "protocol.json", "protocol_sha256": core.sha256(root / "protocol.json"), "decoding": d}
    (root / "frozen.json").write_text(json.dumps(frozen))


def test_confirmation_pairs_fixed_original_and_candidate(tmp_path):
    make_frozen_world(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        probe_model(tmp_path, "model", "data.json", split="test", max_new_tokens=2)
    original = confirm_generation(tmp_path, "frozen.json", "data.json", original=True)
    candidate = confirm_generation(tmp_path, "frozen.json", "data.json")
    assert original["phase"] == candidate["phase"] == "heldout_confirmation"
    assert original["examples"] == candidate["examples"]


@pytest.mark.parametrize("target", ["data.json", "protocol.json", "model/config.json", "identity/replacements.safetensors"])
def test_changed_frozen_inputs_rejected(tmp_path, target):
    make_frozen_world(tmp_path)
    p = tmp_path / target
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|differs"):
        confirm_generation(tmp_path, "frozen.json", "data.json")
