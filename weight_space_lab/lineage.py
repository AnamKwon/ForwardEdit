"""Read-only numeric comparison of successive sparse parameter overlays.

This proves which values changed, not that an edit was feedback-dependent or good.
No model execution, gradient, parameter selection or weight writing occurs here.
"""
from pathlib import Path
import copy
import json
import numpy as np
from safetensors import safe_open
from . import core, evaluation as ev


def revise_candidate(root, model_dir, parent_candidate_dir, operations, hypothesis, rationale, limits, output):
    """Append explicit operations to a verified parent; no choice/fitting/forward."""
    root = Path(root)
    fields = ("observation", "hypothesis", "value_derivation", "prediction", "disconfirming_result")
    if not isinstance(rationale, dict) or any(not isinstance(rationale.get(k), str) or not rationale[k].strip() for k in fields):
        raise ValueError("rationale needs observed feedback, hypothesis, value_derivation, prediction and disconfirming_result")
    json.dumps(rationale, allow_nan=False)
    if not isinstance(operations, list) or not 1 <= len(operations) <= 16:
        raise ValueError("supply1..16 explicit new operations")
    manifest, _ = ev.verify_candidate(root, model_dir, parent_candidate_dir)
    parent_path = core.local(root, parent_candidate_dir)
    parent_sha = core.sha256(parent_path / "replacements.safetensors")
    manifest_sha = core.sha256(parent_path / "manifest.json")
    plan = copy.deepcopy(manifest["plan"])
    by_tensor = {op["tensor"]: op for op in plan["operations"]}
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("kind") == "sequence":
            raise ValueError("revision operations must be explicit non-sequence steps")
        key = operation["tensor"]
        if key not in by_tensor:
            by_tensor[key] = copy.deepcopy(operation)
        else:
            previous = by_tensor[key]
            steps = previous["steps"] if previous["kind"] == "sequence" else [previous]
            by_tensor[key] = {"tensor": key, "kind": "sequence", "steps": [*steps, copy.deepcopy(operation)]}
    plan.update(operations=list(by_tensor.values()), hypothesis=hypothesis, limits=limits)
    if sum(len(op["steps"]) if op["kind"] == "sequence" else 1 for op in plan["operations"]) > 32:
        raise ValueError("revision chain exceeds32 total explicit steps")
    if (core.sha256(parent_path / "replacements.safetensors") != parent_sha
            or core.sha256(parent_path / "manifest.json") != manifest_sha):
        raise ValueError("parent changed during revision")
    return core.write_candidate(root, plan, output, lineage={"kind": "explicit_revision",
        "parent_candidate_dir": parent_candidate_dir, "parent_patch_sha256": parent_sha,
        "parent_manifest_sha256": manifest_sha, "rationale": rationale,
        "interpretation": "Parent retained and explicit new operations appended with dtype rounding each step. Rationale is caller-authored, not verified causality/learning."})


def compare_candidate_parameters(root, model_dir, candidate_dir, parent_candidate_dir=None):
    root = Path(root)
    child_manifest, child = ev.verify_candidate(root, model_dir, candidate_dir)
    parent = {} if parent_candidate_dir is None else ev.verify_candidate(root, model_dir, parent_candidate_dir)[1]
    source = ev._model_dir(root, model_dir) / ev.MODEL_FILE
    source_hash = core.sha256(source)
    child_hash = core.sha256(core.local(root, candidate_dir) / "replacements.safetensors")
    parent_hash = None if parent_candidate_dir is None else core.sha256(
        core.local(root, parent_candidate_dir) / "replacements.safetensors")
    tensors, totals = {}, {k: 0 for k in ("incremental_changed", "net_changed", "retained_parent_changes",
                                          "revised_parent_changes", "restored_to_original", "new_changes")}
    with safe_open(source, framework="np") as handle:
        for key in sorted(set(parent) | set(child)):
            original = handle.get_tensor(key)
            before = parent.get(key, original)
            after = child.get(key, original)
            parent_changed, net_changed = before != original, after != original
            changed = before != after
            counts = {
                "incremental_changed": int(changed.sum()), "net_changed": int(net_changed.sum()),
                "retained_parent_changes": int((parent_changed & ~changed).sum()),
                "revised_parent_changes": int((parent_changed & net_changed & changed).sum()),
                "restored_to_original": int((parent_changed & ~net_changed).sum()),
                "new_changes": int((~parent_changed & net_changed).sum())}
            for name, value in counts.items():
                totals[name] += value
            delta = after.astype(np.float64) - before.astype(np.float64)
            examples = [{"row": int(i), "column": int(j), "original": float(original[i, j]),
                         "parent": float(before[i, j]), "child": float(after[i, j])}
                        for i, j in np.argwhere(changed)[:8]]
            tensors[key] = {**counts, "incremental_l2": float(np.linalg.norm(delta)),
                            "incremental_max_abs": float(np.abs(delta).max()), "examples": examples}
    if (core.sha256(source) != source_hash
            or core.sha256(core.local(root, candidate_dir) / "replacements.safetensors") != child_hash
            or (parent_candidate_dir is not None and core.sha256(
                core.local(root, parent_candidate_dir) / "replacements.safetensors") != parent_hash)):
        raise ValueError("revision inputs changed during comparison")
    return {"source_sha256": source_hash, "parent_candidate_dir": parent_candidate_dir,
            "parent_patch_sha256": parent_hash, "candidate_dir": candidate_dir,
            "patch_sha256": child_hash, **totals, "tensors": tensors,
            "model_forward_calls": 0,
            "interpretation": "Exact parameter lineage only; numeric differences do not prove feedback dependence, causality or learning."}
