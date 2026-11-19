"""Token-level baseline/candidate evaluation of core replacement patches.

torch/transformers are imported lazily, so importing this module never requires
them. Models load from a workspace-relative directory holding ``config.json`` and a
single ``model.safetensors`` (local files only, no remote code, CPU FP32).
Datasets are JSON with explicit token IDs (no tokenizer).

Dataset format::

    {"name": str, "version": str,
     "splits": {"discovery": [{"id": "d0", "tokens": [1, 2, 3]}, ...],
                "validation": [...], "test": [...]}}

Tied weights: a patched tensor that shares storage with another model parameter
(e.g. GPT-2 ``transformer.wte.weight`` / ``lm_head.weight``) is REJECTED with a
clear error; edit an untied tensor instead.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from . import core

__all__ = ["load_dataset", "load_causal_lm", "evaluate_model", "evaluate_baseline",
           "verify_candidate", "apply_candidate", "evaluate_candidate", "write_report"]

MODEL_FILE = "model.safetensors"


def _torch():
    try:
        import torch
        import transformers
    except ImportError as error:  # optional dependency
        raise RuntimeError("evaluation requires torch and transformers") from error
    return torch, transformers


def _file_sha(path: Path) -> str:
    return core.sha256(path)


def _canon_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------- dataset
def load_dataset(root: Path, path: str) -> dict:
    """Load and structurally validate a token dataset. Returns dict with keys
    path, sha256, name, version, splits, split_sha256, cross_split_duplicate_sequences."""
    root = Path(root)
    file = core.local(root, path)
    raw = json.loads(file.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("splits"), dict) or not raw["splits"]:
        raise ValueError("dataset needs a nonempty 'splits' object")
    seen_ids: set = set()
    owner: dict = {}
    duplicates = 0
    splits: dict = {}
    for split, examples in raw["splits"].items():
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"split {split!r} must be a nonempty list")
        clean = []
        for example in examples:
            if not isinstance(example, dict) or not isinstance(example.get("id"), str) or not example["id"]:
                raise ValueError(f"split {split!r}: each example needs a string id")
            if example["id"] in seen_ids:
                raise ValueError(f"duplicate example id {example['id']!r}")
            seen_ids.add(example["id"])
            tokens = example.get("tokens")
            if (not isinstance(tokens, list) or len(tokens) < 2
                    or any(type(t) is not int or t < 0 for t in tokens)):
                raise ValueError(f"example {example['id']!r}: tokens must be >=2 nonnegative ints")
            key = tuple(tokens)
            if key in owner and owner[key] != split:
                duplicates += 1
            owner.setdefault(key, split)
            clean.append({"id": example["id"], "tokens": list(tokens)})
        splits[split] = clean
    return {"path": path, "sha256": _file_sha(file), "name": str(raw.get("name", "")),
            "version": str(raw.get("version", "")), "splits": splits,
            "split_sha256": {k: _canon_hash(v) for k, v in splits.items()},
            "cross_split_duplicate_sequences": duplicates}


# ------------------------------------------------------------------- model
def _model_dir(root: Path, model_dir: str) -> Path:
    directory = core.local(Path(root), model_dir)
    # Validate files as well as their parent: a local directory can contain a
    # symlink to a checkpoint outside the declared workspace.
    for name in ("config.json", MODEL_FILE):
        core.local(Path(root), str(directory / name))
    if not (directory / "config.json").is_file() or not (directory / MODEL_FILE).is_file():
        raise ValueError(f"model dir needs config.json and {MODEL_FILE}")
    for extra in ("model.safetensors.index.json", "pytorch_model.bin"):
        if (directory / extra).exists():
            raise ValueError(f"unsupported extra weight file {extra}")
    return directory


def load_causal_lm(root: Path, model_dir: str, device: str = "cpu"):
    """Fresh CPU FP32 eval-mode AutoModelForCausalLM from local files only."""
    torch, transformers = _torch()
    directory = _model_dir(root, model_dir)
    model, loading = transformers.AutoModelForCausalLM.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, use_safetensors=True, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("incomplete or incompatible checkpoint; refusing randomly initialized parameters")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda (inference only)")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return model.to(device).eval().requires_grad_(False)


def _context_limit(model) -> int:
    config = model.config
    for name in ("n_positions", "max_position_embeddings"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("cannot determine context limit from config")


def evaluate_model(model, dataset: dict, splits=None, batch_size: int = 64) -> dict:
    """Token-weighted next-token NLL (nats) and accuracy per split.

    Returns {split: {examples, tokens, nll, accuracy, per_example: [{id, tokens, nll_sum, correct}]}}."""
    torch, _ = _torch()
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer in [1, 64]")
    if splits is not None and (not isinstance(splits, (list, tuple)) or not splits):
        raise ValueError("splits must be a nonempty list or tuple")
    vocab, context = model.config.vocab_size, _context_limit(model)
    result = {}
    for split in (splits or list(dataset["splits"])):
        if split not in dataset["splits"]:
            raise ValueError(f"unknown split {split!r}")
        examples = dataset["splits"][split]
        for example in examples:
            tokens = example["tokens"]
            if len(tokens) > context:
                raise ValueError(f"example {example['id']!r} exceeds context {context}")
            if max(tokens) >= vocab:
                raise ValueError(f"example {example['id']!r} has token id >= vocab {vocab}")
        rows = [None] * len(examples)
        by_length: dict = {}
        for index, example in enumerate(examples):
            by_length.setdefault(len(example["tokens"]), []).append(index)
        with torch.inference_mode():
            for length, indices in by_length.items():
                for start in range(0, len(indices), batch_size):
                    chunk = indices[start:start + batch_size]
                    batch = torch.tensor([examples[i]["tokens"] for i in chunk], dtype=torch.long, device=next(model.parameters()).device)
                    logits = model(input_ids=batch[:, :-1]).logits.double()
                    if not torch.isfinite(logits).all():
                        raise ValueError("nonfinite logits")
                    logp = torch.log_softmax(logits, dim=-1)
                    target = batch[:, 1:]
                    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    correct = (logits.argmax(-1) == target)
                    for row, i in enumerate(chunk):
                        rows[i] = {"id": examples[i]["id"], "tokens": length - 1,
                                   "nll_sum": float(nll[row].sum()), "correct": int(correct[row].sum())}
        total = sum(r["tokens"] for r in rows)
        nll_sum = sum(r["nll_sum"] for r in rows)
        if total <= 0 or not math.isfinite(nll_sum):
            raise ValueError("nonfinite or empty evaluation")
        result[split] = {"examples": len(rows), "tokens": total, "nll": nll_sum / total,
                         "accuracy": sum(r["correct"] for r in rows) / total, "per_example": rows}
    return result


def _strip(metrics: dict) -> dict:
    return {k: {a: b for a, b in v.items() if a != "per_example"} for k, v in metrics.items()}


def _provenance(root: Path, model_dir: str, dataset: dict) -> dict:
    torch, transformers = _torch()
    directory = _model_dir(root, model_dir)
    return {"model_dir": model_dir, "source_sha256": _file_sha(directory / MODEL_FILE),
            "config_sha256": _file_sha(directory / "config.json"),
            "dataset": {k: dataset[k] for k in ("path", "sha256", "name", "version", "split_sha256",
                                                "cross_split_duplicate_sequences")},
            "evaluator_sha256": _file_sha(Path(__file__)), "dtype": "float32", "device": "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__}


def evaluate_baseline(root: Path, model_dir: str, dataset_path: str, splits=None) -> dict:
    """Unmodified-model metrics + provenance (no per-example rows)."""
    dataset = load_dataset(root, dataset_path)
    metrics = evaluate_model(load_causal_lm(root, model_dir), dataset, splits)
    return {**_provenance(root, model_dir, dataset), "metrics": _strip(metrics)}


# --------------------------------------------------------------- candidates
def verify_candidate(root: Path, model_dir: str, candidate_dir: str):
    """Check hashes/keys/shapes/dtypes/values of a core candidate. Returns (manifest, patch)."""
    root = Path(root)
    cdir = core.local(root, candidate_dir)
    if not cdir.is_dir() or {p.name for p in cdir.iterdir()} != {"replacements.safetensors", "manifest.json"}:
        raise ValueError("candidate dir must contain exactly replacements.safetensors and manifest.json")
    for name in ("manifest.json", "replacements.safetensors"):
        core.local(root, str(cdir / name))
    manifest = json.loads((cdir / "manifest.json").read_text())
    plan, report = manifest.get("plan"), manifest.get("report")
    if not isinstance(plan, dict) or not isinstance(report, dict):
        raise ValueError("manifest needs plan and report")
    model_path = _model_dir(root, model_dir) / MODEL_FILE
    if core.local(root, plan["checkpoint"]) != model_path.resolve():
        raise ValueError("plan checkpoint is not this model directory's model.safetensors")
    source = _file_sha(model_path)
    if not (source == plan["source_sha256"] == report["source_sha256"]):
        raise ValueError("source hash mismatch between model, plan and manifest")
    patch_file = cdir / "replacements.safetensors"
    if _file_sha(patch_file) != report.get("patch_sha256"):
        raise ValueError("patch hash mismatch")
    expected_keys = [op["tensor"] for op in plan["operations"]]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("duplicate tensors in plan")
    patch = {}
    with safe_open(patch_file, framework="np") as handle, safe_open(model_path, framework="np") as src:
        if set(handle.keys()) != set(expected_keys):
            raise ValueError("patch keys differ from plan operations (unrelated tensor injection?)")
        for key in expected_keys:
            if key not in set(src.keys()):
                raise ValueError(f"tensor {key!r} not in source checkpoint")
            if (handle.get_slice(key).get_shape() != src.get_slice(key).get_shape()
                    or handle.get_slice(key).get_dtype() != src.get_slice(key).get_dtype()):
                raise ValueError(f"patch shape/dtype mismatch for {key!r}")
            patch[key] = handle.get_tensor(key)
    # Independent recomputation of the edit also re-enforces plan budgets.
    expected, recomputed = core.prepare(root, plan)
    for key in expected_keys:
        if not np.array_equal(expected[key], patch[key]):
            raise ValueError(f"patch values for {key!r} differ from plan recomputation")
    if recomputed["changed_parameters"] != report.get("changed_parameters"):
        raise ValueError("manifest changed_parameters inconsistent with plan")
    return manifest, patch


def _tied_names(model) -> set:
    groups: dict = {}
    for name, tensor in model.state_dict().items():
        groups.setdefault((tensor.data_ptr(), tuple(tensor.shape)), []).append(name)
    return {n for names in groups.values() if len(names) > 1 for n in names}


def apply_candidate(root: Path, model_dir: str, candidate_dir: str):
    """Fresh model with verified replacements applied in memory. The files are untouched.
    Returns (model, {"manifest", "changed_parameters", "tensors"})."""
    torch, _ = _torch()
    manifest, patch = verify_candidate(root, model_dir, candidate_dir)
    model = load_causal_lm(root, model_dir)
    state, tied = model.state_dict(), _tied_names(model)
    applied = []
    with torch.no_grad():
        for key, array in patch.items():
            if key not in state:
                raise ValueError(f"patched tensor {key!r} is not a model parameter (would be a silent no-op)")
            if key in tied:
                raise ValueError(f"tensor {key!r} is tied/shared with another parameter; "
                                 "editing tied weights is not supported - edit an untied tensor")
            param = state[key]
            value = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))
            if tuple(param.shape) != tuple(value.shape):
                raise ValueError(f"shape mismatch for {key!r}")
            before = param.clone()
            param.copy_(value)
            if not torch.equal(state[key], value):
                raise ValueError(f"patch for {key!r} did not take effect")
            applied.append({"tensor": key, "changed_parameters": int((param != before).sum())})
    total = sum(a["changed_parameters"] for a in applied)
    if total != manifest["report"]["changed_parameters"]:
        raise ValueError("applied change count differs from manifest (silent no-op or mismatch)")
    return model, {"manifest": manifest, "changed_parameters": total, "tensors": applied}


def _paired(base_rows, cand_rows, seed: int, draws: int) -> dict:
    b = np.array([r["nll_sum"] for r in base_rows])
    c = np.array([r["nll_sum"] for r in cand_rows])
    t = np.array([r["tokens"] for r in base_rows], dtype=float)
    gain = float((b.sum() - c.sum()) / t.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(draws, len(b)))
    boot = (b[idx].sum(1) - c[idx].sum(1)) / t[idx].sum(1)
    worse = [(float((cj - bj) / tj), r["id"]) for bj, cj, tj, r in zip(b, c, t, base_rows) if cj > bj]
    worse.sort(reverse=True)
    return {"nll_gain": gain, "gain_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "regressed_examples": len(worse), "worst_regressions": [{"id": i, "nll_increase_per_token": v}
                                                                   for v, i in worse[:5]]}


def evaluate_candidate(root: Path, model_dir: str, candidate_dir: str, dataset_path: str, *,
                       min_nll_gain: float, selection_split: str = "validation",
                       heldout_split: str = "test", preserve_splits=(),
                       max_nll_regression: float = 0.0, max_accuracy_drop: float = 0.0,
                       mode: str = "direct_edit", seed: int = 0, bootstrap: int = 1000) -> dict:
    """Paired baseline-vs-candidate evaluation with a strict pre-declared gate.

    Gate passes only if NLL gain (baseline-candidate, nats/token) is STRICTLY greater than
    min_nll_gain (>0, set before looking at heldout) on selection_split and heldout_split,
    the heldout bootstrap 95% CI lower bound is > 0, heldout accuracy drop <= max_accuracy_drop,
    every preserve_split NLL increase <= max_nll_regression, and no heldout sequence appears in
    another split. The report never applies or overwrites anything ('adoption' stays advisory)."""
    root = Path(root)
    if not (isinstance(min_nll_gain, (int, float)) and math.isfinite(min_nll_gain) and min_nll_gain > 0):
        raise ValueError("min_nll_gain must be a finite positive number declared in advance")
    if selection_split == heldout_split:
        raise ValueError("selection and heldout splits must differ")
    for name, value in (("max_nll_regression", max_nll_regression), ("max_accuracy_drop", max_accuracy_drop)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if max_accuracy_drop > 1:
        raise ValueError("max_accuracy_drop must be <= 1")
    if type(bootstrap) is not int or not 1 <= bootstrap <= 10000:
        raise ValueError("bootstrap must be an integer in [1, 10000]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if mode not in ("direct_edit", "gradient_free_search"):
        raise ValueError("only gradient-free direct editing is allowed")
    dataset = load_dataset(root, dataset_path)
    wanted = [selection_split, heldout_split, *[s for s in preserve_splits]]
    for split in wanted:
        if split not in dataset["splits"]:
            raise ValueError(f"unknown split {split!r}")
    names = list(dict.fromkeys(wanted + list(dataset["splits"])))
    source_file = _model_dir(root, model_dir) / MODEL_FILE
    source_before = _file_sha(source_file)
    base = evaluate_model(load_causal_lm(root, model_dir), dataset, names)
    model, info = apply_candidate(root, model_dir, candidate_dir)
    cand = evaluate_model(model, dataset, names)
    if _file_sha(source_file) != source_before:
        raise ValueError("source checkpoint changed during evaluation")
    comparison = {}
    for split in names:
        paired = _paired(base[split]["per_example"], cand[split]["per_example"], seed, bootstrap)
        paired["accuracy_change"] = cand[split]["accuracy"] - base[split]["accuracy"]
        comparison[split] = paired
    held_owner = {tuple(e["tokens"]) for s, ex in dataset["splits"].items() if s != heldout_split for e in ex}
    leakage = sum(tuple(e["tokens"]) in held_owner for e in dataset["splits"][heldout_split])
    checks = {
        "selection_gain_exceeds_min": comparison[selection_split]["nll_gain"] > min_nll_gain,
        "heldout_gain_exceeds_min": comparison[heldout_split]["nll_gain"] > min_nll_gain,
        "heldout_ci_lower_positive": comparison[heldout_split]["gain_ci95"][0] > 0,
        "heldout_accuracy_drop_ok": comparison[heldout_split]["accuracy_change"] >= -max_accuracy_drop,
        "preserve_splits_ok": all(comparison[s]["nll_gain"] >= -max_nll_regression for s in preserve_splits),
        "no_heldout_leakage": leakage == 0,
    }
    manifest = info["manifest"]
    return {**_provenance(root, model_dir, dataset), "mode": mode, "seed": seed, "bootstrap_draws": bootstrap,
            "candidate_dir": candidate_dir, "patch_sha256": manifest["report"]["patch_sha256"],
            "hypothesis": manifest["plan"]["hypothesis"], "operations": manifest["plan"]["operations"],
            "edit": {"changed_parameters": info["changed_parameters"],
                     "changed_fraction": manifest["report"]["changed_fraction"],
                     "tensors": manifest["report"]["tensors"]},
            "numeric_validation": "verified", "baseline": _strip(base), "candidate": _strip(cand),
            "comparison": comparison,
            "gate": {"min_nll_gain": min_nll_gain, "selection_split": selection_split,
                     "heldout_split": heldout_split, "preserve_splits": list(preserve_splits),
                     "max_nll_regression": max_nll_regression, "max_accuracy_drop": max_accuracy_drop,
                     "checks": checks, "passed": all(checks.values())},
            "behavior_validation": "gate_passed" if all(checks.values()) else "gate_failed",
            "adoption": "recommended_not_applied" if all(checks.values()) else "rejected",
            "original_modified": False}


def write_report(root: Path, report: dict, output: str) -> str:
    """Write report JSON to a new workspace-relative path; refuses to overwrite."""
    target = core.local(Path(root), output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(report, stream, indent=2)
    return output
