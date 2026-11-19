"""Frozen-choice heldout generation. This is workflow integrity, not an OS sandbox."""
import json
from pathlib import Path
from . import core, evaluation as ev


def validate_confirmation(root, frozen_path, model_dir, dataset_path, candidate_dir,
                          max_new_tokens, stop_at_eos, use_cache, device):
    root = Path(root)
    path = core.local(root, frozen_path)
    frozen = json.loads(path.read_text())
    if frozen.get("kind") != "frozen_generation_choice" or not frozen.get("selected_candidate"):
        raise ValueError("a frozen selected generation candidate is required")
    protocol_path = core.local(root, frozen["protocol_path"])
    if core.sha256(protocol_path) != frozen["protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if frozen["decoding"] != protocol["decoding"]:
        raise ValueError("decoding differs from original protocol")
    decoding = frozen["decoding"]
    if (max_new_tokens != decoding["max_new_tokens"] or stop_at_eos != decoding["stop_at_eos"]
            or use_cache != decoding["use_cache"] or device != decoding["device"]):
        raise ValueError("heldout decoding must match frozen decoding")
    directory = ev._model_dir(root, model_dir)
    if (model_dir != frozen["model_dir"]
            or core.sha256(directory / ev.MODEL_FILE) != frozen["source_sha256"]
            or core.sha256(directory / "config.json") != frozen["config_sha256"]):
        raise ValueError("frozen original model changed")
    selected = frozen["selected_candidate"]
    patch = core.local(root, selected) / "replacements.safetensors"
    if core.sha256(patch) != frozen["patch_sha256"]:
        raise ValueError("frozen candidate patch changed")
    if candidate_dir not in (None, selected):
        raise ValueError("only original or frozen selected candidate can be confirmed")
    declared = protocol["sealed_files"]
    if dataset_path not in declared or core.sha256(core.local(root, dataset_path)) != declared[dataset_path]:
        raise ValueError("heldout dataset differs from initially sealed data")
    return {"frozen_path": frozen_path, "frozen_sha256": core.sha256(path)}


def confirm_generation(root, frozen_path, dataset_path, original=False):
    if type(original) is not bool:
        raise ValueError("original must be boolean")
    frozen = json.loads(core.local(Path(root), frozen_path).read_text())
    d = frozen["decoding"]
    from .inference import probe_model
    return probe_model(root, frozen["model_dir"], dataset_path,
        candidate_dir=None if original else frozen["selected_candidate"], split="test", max_examples=8,
        top_k=5, max_new_tokens=d["max_new_tokens"], device=d["device"],
        stop_at_eos=d["stop_at_eos"], use_cache=d["use_cache"], _confirmation_path=frozen_path)
