"""Host/LLM-authored bounded plan search; no provider calls or optimizer."""
import copy
import json
import math
from pathlib import Path
from . import core, evaluation as ev


def _finite(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def run_experiment(root, config, output):
    """Config: model_dir, dataset_path, plans (1..12 explicit core plans), min_nll_gain,
    optional selection_split, heldout_split, preserve_splits, max_nll_regression,
    max_accuracy_drop, seed, bootstrap, max_examples and max_tokens (per search split).
    Search permits discovery/validation only. Output is reserved once, never reused.
    """
    root = Path(root)
    allowed = {"model_dir", "dataset_path", "plans", "min_nll_gain", "selection_split", "heldout_split",
               "preserve_splits", "max_nll_regression", "max_accuracy_drop", "seed", "bootstrap",
               "max_examples", "max_tokens", "design_mode"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unknown experiment config fields")
    design_mode = config.get("design_mode", "direct_edit")
    if design_mode != "direct_edit":
        raise ValueError("only direct_edit is allowed; gradients and training are disabled")
    search_mode = "gradient_free_search"
    plans = config.get("plans")
    if not isinstance(plans, list) or not 1 <= len(plans) <= 12:
        raise ValueError("experiment requires 1..12 LLM-authored plans")
    gain = _finite(config.get("min_nll_gain"), "min_nll_gain", True)
    regression = _finite(config.get("max_nll_regression", 0), "max_nll_regression")
    accuracy = _finite(config.get("max_accuracy_drop", 0), "max_accuracy_drop")
    if accuracy > 1:
        raise ValueError("max_accuracy_drop must be <=1")
    selection, heldout = config.get("selection_split", "validation"), config.get("heldout_split", "test")
    preserve = config.get("preserve_splits", [])
    if (selection not in ("discovery", "validation") or heldout in ("discovery", "validation")
            or not isinstance(preserve, list) or any(s not in ("discovery", "validation") for s in preserve)
            or len(set(preserve)) != len(preserve)):
        raise ValueError("search/preserve splits must be discovery/validation; heldout must be separate")
    seed, bootstrap = config.get("seed", 0), config.get("bootstrap", 1000)
    if type(seed) is not int or seed < 0 or type(bootstrap) is not int or not 1 <= bootstrap <= 10000:
        raise ValueError("invalid seed/bootstrap budget")
    max_examples, max_tokens = config.get("max_examples", 256), config.get("max_tokens", 32768)
    if (type(max_examples) is not int or not 1 <= max_examples <= 1024
            or type(max_tokens) is not int or not 1 <= max_tokens <= 131072):
        raise ValueError("invalid example/token budget")
    model_dir, dataset_path = config["model_dir"], config["dataset_path"]
    model_file = ev._model_dir(root, model_dir) / ev.MODEL_FILE
    source = core.sha256(model_file)
    dataset = ev.load_dataset(root, dataset_path)
    wanted = list(dict.fromkeys([selection, *preserve]))
    for split in [*wanted, heldout]:
        if split not in dataset["splits"]:
            raise ValueError("missing split")
    # Bounds include confirmation, whose protected API evaluates every dataset split.
    for examples in dataset["splits"].values():
        if len(examples) > max_examples or sum(len(e["tokens"]) for e in examples) > max_tokens:
            raise ValueError("dataset exceeds declared per-split evaluation budget")
    for plan in plans:
        if core.local(root, plan["checkpoint"]) != model_file.resolve() or plan["source_sha256"] != source:
            raise ValueError("all plans must branch from shared original source")
        core.prepare(root, plan)
    target = core.local(root, output)
    target.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        with (target / name).open("x") as stream:
            json.dump(obj, stream, indent=2, allow_nan=False)
    save("config.json", config)
    base = ev.evaluate_model(ev.load_causal_lm(root, model_dir), dataset, wanted)
    save("baseline.json", {**ev._provenance(root, model_dir, dataset), "metrics": ev._strip(base)})
    identity = copy.deepcopy(plans[0])
    identity["hypothesis"] = "Independent identity control"
    for op in identity["operations"]:
        if op["kind"] == "set_values":
            op["values"] = copy.deepcopy(op["expected_values"])
        else:
            op["strength"] = 0
    identity_path = str((target / "identity").relative_to(root.resolve()))
    core.write_candidate(root, identity, identity_path)
    control_model, _ = ev.apply_candidate(root, model_dir, identity_path)
    control = ev.evaluate_model(control_model, dataset, wanted)
    identity_ok = control == base
    save("identity.json", {"metrics": ev._strip(control), "matches_baseline": identity_ok})
    if not identity_ok:
        raise ValueError("identity control differs from baseline")
    rows, winner = [], None
    for i, plan in enumerate(plans):
        candidate = str((target / f"candidate-{i:02d}").relative_to(root.resolve()))
        report = core.write_candidate(root, plan, candidate)
        model, _ = ev.apply_candidate(root, model_dir, candidate)
        metrics = ev.evaluate_model(model, dataset, wanted)
        comparison = {s: {"nll_gain": base[s]["nll"] - metrics[s]["nll"],
                          "accuracy_change": metrics[s]["accuracy"] - base[s]["accuracy"]} for s in wanted}
        passed = (comparison[selection]["nll_gain"] > gain
                  and all(comparison[s]["nll_gain"] >= -regression for s in preserve)
                  and all(comparison[s]["accuracy_change"] >= -accuracy for s in wanted))
        row = {"index": i, "candidate_dir": candidate, "patch_sha256": report["patch_sha256"],
               "plan_sha256": ev._canon_hash(plan), "metrics": ev._strip(metrics),
               "comparison": comparison, "selection_passed": passed}
        rows.append(row)
        save(f"selection-{i:02d}.json", row)
        if passed and (winner is None or comparison[selection]["nll_gain"] > winner["comparison"][selection]["nll_gain"]):
            winner = row
    frozen = {"winner": winner, "source_sha256": source, "dataset_sha256": dataset["sha256"],
              "config_sha256": ev._canon_hash(config), "selection_split": selection,
              "heldout_split": heldout, "confirmation_calls": 0 if winner is None else 1, "design_mode": design_mode}
    # Persist the decision before any heldout inference.
    save("frozen-choice.json", frozen)
    confirmation = None
    if winner is not None:
        manifest, _ = ev.verify_candidate(root, model_dir, winner["candidate_dir"])
        if (ev._canon_hash(manifest["plan"]) != winner["plan_sha256"]
                or manifest["report"]["patch_sha256"] != winner["patch_sha256"]
                or core.sha256(core.local(root, dataset_path)) != dataset["sha256"]):
            raise ValueError("frozen inputs changed")
        confirmation = ev.evaluate_candidate(root, model_dir, winner["candidate_dir"], dataset_path,
            min_nll_gain=gain, selection_split=selection, heldout_split=heldout,
            preserve_splits=preserve, max_nll_regression=regression, max_accuracy_drop=accuracy,
            mode="gradient_free_search",
            seed=seed, bootstrap=bootstrap)
        confirmation["design_mode"] = design_mode
        confirmation["search_mode"] = search_mode
        save("confirmation.json", confirmation)
    if core.sha256(model_file) != source:
        raise ValueError("source changed during experiment")
    rejected = [r["index"] for r in rows if winner is None or r["index"] != winner["index"]]
    if winner is not None and confirmation["behavior_validation"] != "gate_passed":
        rejected.append(winner["index"])
    result = {"output": output, "mode": search_mode, "design_mode": design_mode, "plan_author": "external LLM/host",
              "status": "no_winner" if winner is None else confirmation["behavior_validation"],
              "winner": None if winner is None else winner["index"],
              "rejected_candidates": sorted(rejected),
              "confirmation_calls": 0 if winner is None else 1, "original_modified": False}
    save("summary.json", result)
    return result
