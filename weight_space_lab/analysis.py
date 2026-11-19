"""Pre-edit evidence from signed, position-specific affine contributions.

All observations use frozen forward inference. Fixed-input matrix arithmetic is
not a gradient or a parameter solver. Human/LLM labels and semantic hypotheses
remain claims; completeness and local algebra do not establish causation.
"""
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from . import core, evaluation as ev
from .inference import load_inference_model


def analyze_parameter_contexts(root, model_dir, dataset_path, tensor, rows, columns,
                               pairs, output, proposed_values=None, device="cpu", candidate_dir=None):
    """Analyze at most8 declared reference/failure pairs at explicit token positions.

    Each pair contains id, reference_id, failure_id, reference_positions and
    failure_positions (equal-length lists of1..8 zero-based prediction positions).
    Only discovery examples are used. Labels do not certify correct/failed behavior.
    Proposed values are LLM-supplied; this function never chooses or writes weights.
    """
    root = Path(root)
    torch, _ = ev._torch()
    from transformers.pytorch_utils import Conv1D
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 8:
        raise ValueError("provide 1..8 explicit reference/failure pairs")
    ds = ev.load_dataset(root, dataset_path)
    if "discovery" not in ds["splits"]:
        raise ValueError("parameter analysis requires discovery examples")
    examples = {e["id"]: e for e in ds["splits"]["discovery"]}
    pair_ids, needed = set(), set()
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("id"), str) or not pair["id"]:
            raise ValueError("each pair needs a string id")
        if pair["id"] in pair_ids:
            raise ValueError("pair ids must be unique")
        pair_ids.add(pair["id"])
        for role in ("reference", "failure"):
            eid, positions = pair.get(role + "_id"), pair.get(role + "_positions")
            if not isinstance(eid, str) or eid not in examples:
                raise ValueError("pairs must refer only to discovery example ids")
            if not isinstance(positions, list) or not 1 <= len(positions) <= 8:
                raise ValueError("select 1..8 prediction positions per example")
            core._indices(positions, len(examples[eid]["tokens"]) - 1)
            needed.add(eid)
        if len(pair["reference_positions"]) != len(pair["failure_positions"]):
            raise ValueError("reference/failure positions must have equal counts")
    if (any(len(examples[eid]["tokens"]) > 512 for eid in needed)
            or sum(len(examples[eid]["tokens"]) for eid in needed) > 8192):
        raise ValueError("analysis permits <=512 tokens/example and <=8192 total")
    model = load_inference_model(root, model_dir, candidate_dir=candidate_dir, device=device)
    patch_path = None if candidate_dir is None else core.local(root, candidate_dir) / "replacements.safetensors"
    patch_hash = None if patch_path is None else core.sha256(patch_path)
    owner, _, leaf = tensor.rpartition(".")
    module = dict(model.named_modules()).get(owner)
    if leaf != "weight" or not isinstance(module, (torch.nn.Linear, Conv1D)):
        raise ValueError("analyze a Linear or Conv1D weight")
    if tensor in ev._tied_names(model):
        raise ValueError("tied/shared parameters are not supported edit targets")
    weight = model.state_dict()[tensor]
    core._indices(rows, weight.shape[0]); core._indices(columns, weight.shape[1])
    if len(rows) * len(columns) > 1024 or max(len(rows), len(columns)) > 64:
        raise ValueError("analyze <=1024 values and <=64 coordinates per axis")
    linear = isinstance(module, torch.nn.Linear)
    input_ids, output_ids = (columns, rows) if linear else (rows, columns)
    original = weight.detach().cpu().double().numpy()[np.ix_(rows, columns)]
    directory = ev._model_dir(root, model_dir)
    with safe_open(directory / ev.MODEL_FILE, framework="np") as stored:
        source_dtype = stored.get_tensor(tensor).dtype
    if source_dtype != np.dtype("float32"):
        raise ValueError("position-specific edit analysis currently requires an FP32 source tensor")
    proposed = None
    if proposed_values is not None:
        proposed = np.asarray(proposed_values, dtype=np.float64)
        if proposed.shape != original.shape or not np.isfinite(proposed).all():
            raise ValueError("proposed_values must be finite and match rows x columns")
        # Match the actual CPU FP32 overlay used by the inference adapter.
        proposed = proposed.astype(np.float32).astype(np.float64)
        if not np.isfinite(proposed).all():
            raise ValueError("proposed_values overflow inference dtype")
    before = {"source_sha256": core.sha256(directory / ev.MODEL_FILE),
              "config_sha256": core.sha256(directory / "config.json"), "dataset_sha256": ds["sha256"]}
    captures, logits_by_id = {}, {}
    capture = []

    def hook(mod, args, result):
        if args[0].ndim != 3 or result.ndim != 3 or args[0].shape[0] != 1:
            raise ValueError("selected affine module must use batch x token x feature tensors")
        x = args[0][0].detach().double().cpu().numpy()[:, input_ids]
        y = result[0].detach().double().cpu().numpy()[:, output_ids]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite activation")
        capture.append((x, y))

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for eid in sorted(needed):
                tokens = examples[eid]["tokens"]
                if len(tokens) > ev._context_limit(model) or max(tokens) >= model.config.vocab_size:
                    raise ValueError("example exceeds model context or vocabulary")
                capture.clear()
                result = model(input_ids=torch.tensor([tokens[:-1]], device=device))
                if len(capture) != 1:
                    raise ValueError("selected module must execute exactly once per example")
                captures[eid] = capture[0]
                logp = result.logits[0].double().log_softmax(-1)
                if not torch.isfinite(logp).all():
                    raise ValueError("nonfinite model output")
                target = torch.tensor(tokens[1:], device=device)
                logits_by_id[eid] = logp.gather(-1, target[:, None]).squeeze(-1).cpu().tolist()
    finally:
        handle.remove()

    def observation(eid, position):
        x, y = captures[eid]
        contribution = original @ x[position] if linear else x[position] @ original
        entry = {"example_id": eid, "position": position,
                 "prefix_sha256": ev._canon_hash(examples[eid]["tokens"][:position + 1]),
                 "input_token": examples[eid]["tokens"][position],
                 "target_token": examples[eid]["tokens"][position + 1],
                 "target_log_probability": logits_by_id[eid][position],
                 "signed_input": x[position].tolist(), "actual_module_output": y[position].tolist(),
                 "selected_weight_contribution": contribution.tolist()}
        if proposed is not None:
            delta = proposed - original
            entry["fixed_input_output_delta"] = (delta @ x[position] if linear else x[position] @ delta).tolist()
        return entry

    observations = []
    for pair in pairs:
        for rp, fp in zip(pair["reference_positions"], pair["failure_positions"]):
            r = observation(pair["reference_id"], rp)
            f = observation(pair["failure_id"], fp)
            prefix_r = examples[pair["reference_id"]]["tokens"][:rp + 1]
            prefix_f = examples[pair["failure_id"]]["tokens"][:fp + 1]
            shared = 0
            for a, b in zip(prefix_r, prefix_f):
                if a != b:
                    break
                shared += 1
            observations.append({"pair_id": pair["id"], "reference": r, "failure": f,
                "shared_prefix_tokens": shared,
                "identical_prefix": r["prefix_sha256"] == f["prefix_sha256"],
                "signed_input_contrast": (np.asarray(f["signed_input"]) - r["signed_input"]).tolist(),
                "contribution_contrast": (np.asarray(f["selected_weight_contribution"])
                                          - r["selected_weight_contribution"]).tolist()})
    if (core.sha256(directory / ev.MODEL_FILE) != before["source_sha256"]
            or core.sha256(directory / "config.json") != before["config_sha256"]
            or core.sha256(core.local(root, dataset_path)) != before["dataset_sha256"]
            or (patch_path is not None and core.sha256(patch_path) != patch_hash)):
        raise ValueError("analysis inputs changed")
    # Adjacent positions and repeated caller labels are not independent contexts.
    grouped = {}
    for obs in observations:
        r, f = obs["reference"], obs["failure"]
        key = tuple(ev._canon_hash(examples[x["example_id"]]["tokens"]) for x in (r, f))
        group = grouped.setdefault(key, {"reference_ids": set(), "failure_ids": set(), "positions": {}})
        group["reference_ids"].add(r["example_id"])
        group["failure_ids"].add(f["example_id"])
        group["positions"].setdefault((r["position"], f["position"]), obs)
    contexts = []
    for (rhash, fhash), group in grouped.items():
        unique = list(group["positions"].values())
        contexts.append({"reference_ids": sorted(group["reference_ids"]), "failure_ids": sorted(group["failure_ids"]),
            "reference_sequence_sha256": rhash, "failure_sequence_sha256": fhash,
            "unique_position_pairs": len(unique),
            "mean_reference_signed_input": np.mean([o["reference"]["signed_input"] for o in unique], axis=0).tolist(),
            "mean_failure_signed_input": np.mean([o["failure"]["signed_input"] for o in unique], axis=0).tolist(),
            "mean_contribution_contrast": np.mean([o["contribution_contrast"] for o in unique], axis=0).tolist()})
    context_evidence = {"observation_rows": len(observations), "distinct_sequence_pairs": len(contexts),
        "unique_position_pairs": sum(c["unique_position_pairs"] for c in contexts), "contexts": contexts,
        "independence_status": "Distinct token sequences/positions only; statistical independence, semantic correctness and unseen-prompt transfer are not established."}
    report = {"kind": "parameter_context_analysis", "schema_version": 1, **before,
        "model_dir": model_dir, "dataset_path": dataset_path, "split": "discovery",
        "candidate_dir": candidate_dir, "patch_sha256": patch_hash,
        "tensor": tensor, "owner": owner, "module_class": type(module).__name__,
        "layout": "out,in" if linear else "in,out", "input_feature_indices": input_ids,
        "output_feature_indices": output_ids, "rows": rows, "columns": columns,
        "original_values": original.tolist(), "proposed_values": None if proposed is None else proposed.tolist(),
        "observations": observations, "context_evidence": context_evidence,
        "device": device, "gradients_enabled": False, "optimizer_steps": 0,
        "evidence_status": "observational_only", "labels_status": "caller_declared_not_verified",
        "interpretation": "Contribution is x times selected weights only; bias and other features remain. "
            "Proposed delta is exact affine algebra at recorded original input, not a derivative, model-wide "
            "prediction or causal proof. Pair labels and semantic roles require independent evidence."}
    ev.write_report(root, report, output)
    return {"analysis_path": output, "analysis_sha256": core.sha256(core.local(root, output)), "report": report}


def prepare_edit_proposal(root, analysis_path, basis, limits, output):
    """Bind an LLM rationale to recorded numbers; allow only a causal probe, never certify truth.

    The LLM supplies both proposed values and a desired signed output direction.
    No function here chooses values, minimizes loss, or approves deployment.
    """
    root = Path(root)
    path = core.local(root, analysis_path)
    analysis = json.loads(path.read_text())
    if analysis.get("kind") != "parameter_context_analysis" or analysis.get("schema_version") != 1:
        raise ValueError("expected a parameter_context_analysis report")
    if analysis.get("candidate_dir") is not None:
        raise ValueError("prepare proposals from original analysis; edited analysis is intervention evidence only")
    directory = ev._model_dir(root, analysis["model_dir"])
    if (core.sha256(directory / ev.MODEL_FILE) != analysis["source_sha256"]
            or core.sha256(directory / "config.json") != analysis["config_sha256"]
            or core.sha256(core.local(root, analysis["dataset_path"])) != analysis["dataset_sha256"]):
        raise ValueError("analysis is stale; source/config/dataset changed")
    fields = ("failure_description", "structural_role", "mechanism_hypothesis", "pairing_justification", "value_derivation",
              "alternative_explanation", "expected_behavior", "disconfirming_result")
    if not isinstance(basis, dict):
        raise ValueError("basis must be an object")
    # A report should never serialize NaN/Infinity disguised as rationale metadata.
    json.dumps(basis, allow_nan=False)
    missing = [k for k in fields if not isinstance(basis.get(k), str) or not basis[k].strip()]
    reasons = ["missing explanation: " + k for k in missing]
    controls = basis.get("controls", [])
    if not isinstance(controls, list) or not {"identity", "opposing_edit"}.issubset(
            {c.get("kind") for c in controls if isinstance(c, dict) and isinstance(c.get("kind"), str)
             and isinstance(c.get("purpose"), str)
             and c["purpose"].strip()}):
        reasons.append("declare identity and opposing_edit controls with purposes")
    proposed = analysis["proposed_values"]
    if proposed is None:
        reasons.append("no explicit proposed_values observed in analysis")
    pair_ids = {p["pair_id"] for p in analysis["observations"]}
    cited = basis.get("evidence_pairs", [])
    if (not isinstance(cited, list) or not cited or any(not isinstance(p, str) for p in cited)
            or not set(cited).issubset(pair_ids)):
        reasons.append("cite existing evidence_pairs from this analysis")
    rule = basis.get("value_rule", {})
    if proposed is not None:
        old, new = np.asarray(analysis["original_values"]), np.asarray(proposed)
        if isinstance(rule, dict) and rule.get("kind") == "attenuate":
            fraction = rule.get("fraction")
            if (type(fraction) not in (float, int) or not math.isfinite(fraction) or not 0 < fraction <= 1
                    or not np.array_equal(new, (old * (1 - fraction)).astype(np.float32).astype(np.float64))):
                reasons.append("proposed values do not match declared attenuation arithmetic")
        elif isinstance(rule, dict) and rule.get("kind") == "explicit":
            entries = rule.get("coordinate_reasons", [])
            explained = {(e.get("row"), e.get("column")) for e in entries if isinstance(e, dict)
                and type(e.get("row")) is int and type(e.get("column")) is int
                and isinstance(e.get("reason"), str) and e["reason"].strip()} if isinstance(entries, list) else set()
            changed = {(analysis["rows"][i], analysis["columns"][j])
                       for i, j in zip(*np.nonzero(old != new))}
            if not changed or explained != changed:
                reasons.append("explicit values need a coordinate reason for exactly every changed weight")
        elif isinstance(rule, dict) and rule.get("kind") == "rank_one":
            left = np.asarray(rule.get("left", []), dtype=np.float64)
            right = np.asarray(rule.get("right", []), dtype=np.float64)
            strength = rule.get("strength")
            valid = (left.shape == (old.shape[0],) and right.shape == (old.shape[1],)
                     and np.isfinite(left).all() and np.isfinite(right).all()
                     and type(strength) in (int, float) and math.isfinite(strength) and strength >= 0)
            if not valid or not np.array_equal(new, (old + strength * np.outer(left, right)).astype(np.float32).astype(np.float64)):
                reasons.append("proposed values do not match declared rank-one arithmetic")
        else:
            reasons.append("declare value_rule: attenuate, explicit with coordinate_reasons, or rank_one with vectors/strength")
    direction = np.asarray(basis.get("target_output_direction", []), dtype=np.float64)
    expected_size = len(analysis["output_feature_indices"])
    norm = np.linalg.norm(direction)
    direction_valid = direction.shape == (expected_size,) and np.isfinite(direction).all() and np.isfinite(norm) and norm > 0
    bound = basis.get("max_reference_output_delta")
    bound_valid = type(bound) in (int, float) and math.isfinite(bound) and bound >= 0
    if not direction_valid:
        reasons.append("supply a finite nonzero target_output_direction in selected output coordinates")
    if not bound_valid:
        reasons.append("declare finite nonnegative max_reference_output_delta")
    checks, numeric = {}, None
    if proposed is not None and direction_valid and bound_valid:
        unit = direction / norm
        reference = np.asarray([p["reference"]["fixed_input_output_delta"] for p in analysis["observations"]])
        failure = np.asarray([p["failure"]["fixed_input_output_delta"] for p in analysis["observations"]])
        projected_r, projected_f = reference @ unit, failure @ unit
        checks = {"failure_change_aligned": bool(projected_f.mean() > 0),
                  "failure_more_affected_than_reference": bool(projected_f.mean() > projected_r.mean()),
                  "reference_within_declared_bound": bool(np.linalg.norm(reference, axis=1).max() <= bound),
                  "nonidentical_contexts": all(not p["identical_prefix"] for p in analysis["observations"])}
        numeric = {"mean_reference_projected_delta": float(projected_r.mean()),
                   "mean_failure_projected_delta": float(projected_f.mean()),
                   "max_reference_output_delta_norm": float(np.linalg.norm(reference, axis=1).max()),
                   "fixed_input_only": True}
        reasons.extend("local check failed: " + k for k, passed in checks.items() if not passed)
    plan = None
    if proposed is not None:
        plan = {"checkpoint": str((directory / ev.MODEL_FILE).relative_to(root.resolve())),
            "source_sha256": analysis["source_sha256"], "hypothesis": basis.get("mechanism_hypothesis", ""),
            "limits": limits, "operations": [{"kind": "set_values", "tensor": analysis["tensor"],
                "rows": analysis["rows"], "columns": analysis["columns"],
                "expected_values": analysis["original_values"], "values": proposed}]}
        if not missing:
            try:
                core.prepare(root, plan)
            except (ValueError, KeyError, TypeError) as error:
                reasons.append("numeric edit validation: " + str(error))
    report = {"kind": "edit_basis_review", "analysis_path": analysis_path,
        "analysis_sha256": core.sha256(path), "source_sha256": analysis["source_sha256"],
        "basis": basis, "local_checks": checks, "local_prediction": numeric,
        "decision": "defer" if reasons else "ready_for_causal_probe", "reasons": reasons,
        "plan": None if reasons else plan,
        "causal_status": "unverified", "performance_status": "unverified",
        "interpretation": "Checks establish explanation completeness and fixed-input affine consistency only. "
            "Text is an LLM claim, not verified causation. Bounds are declared proposal criteria, not evidence "
            "of preregistration. A real controlled intervention and independent validation are still required.",
        "original_modified": False, "candidate_created": False}
    ev.write_report(root, report, output)
    return report
