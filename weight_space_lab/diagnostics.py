"""Bounded real-forward evidence, never an optimizer or semantic classifier."""
from pathlib import Path
import numpy as np
from . import core, evaluation as ev


def _top(values, k):
    a = np.asarray(values, dtype=float)
    return [{"index": int(i), "value": float(a[i])}
            for i in np.argsort(-np.abs(a), kind="stable")[:k]]


def diagnose_model(root, model_dir, dataset_path, tensors, split="discovery",
                   max_examples=8, top_k=8, include_gradients=False, candidate_dir=None, device="cpu"):
    """Linear/Conv1D input/output feature evidence on at most 32 examples/8 matrices.

    Full per-feature summaries are bounded to 4096 features; matrices to 4M entries.
    Gradient requests are rejected; this function only observes inference.
    """
    if (type(max_examples) is not int or not 1 <= max_examples <= 32
            or type(top_k) is not int or not 1 <= top_k <= 32):
        raise ValueError("max_examples must be 1..32 and top_k 1..32")
    if (not isinstance(tensors, list) or not 1 <= len(tensors) <= 8
            or any(not isinstance(t, str) or not t for t in tensors)
            or len(set(tensors)) != len(tensors)):
        raise ValueError("request 1..8 unique matrix names")
    if include_gradients:
        raise ValueError("gradient diagnostics are disabled: inference-only direct editing")
    root = Path(root)
    directory = ev._model_dir(root, model_dir)
    before = core.sha256(directory / ev.MODEL_FILE)
    config_before = core.sha256(directory / "config.json")
    dataset = ev.load_dataset(root, dataset_path)
    if split not in dataset["splits"]:
        raise ValueError("unknown split")
    examples = dataset["splits"][split][:max_examples]
    if sum(len(e["tokens"]) for e in examples) > 8192:
        raise ValueError("diagnostic token budget exceeded (8192)")
    sampled = {**dataset, "splits": {split: examples}}
    torch, _ = ev._torch()
    from transformers.pytorch_utils import Conv1D
    from .inference import load_inference_model
    model = load_inference_model(root, model_dir, candidate_dir, device)
    modules, params = dict(model.named_modules()), dict(model.named_parameters())
    tied = ev._tied_names(model)
    records, hooks = {}, []
    for name in tensors:
        owner, _, leaf = name.rpartition(".")
        module = modules.get(owner)
        if name in tied:
            raise ValueError("tied/shared parameters unsupported")
        if leaf != "weight" or name not in params or not isinstance(module, (torch.nn.Linear, Conv1D)):
            raise ValueError("selected weight must belong to Linear or Conv1D")
        w = params[name]
        if w.ndim != 2 or w.numel() > 4_000_000 or max(w.shape) > 4096:
            raise ValueError("diagnostic matrix budget exceeded")
        if not torch.isfinite(w).all():
            raise ValueError("nonfinite weight")
        records[name] = {"module": owner, "module_class": type(module).__name__,
                         "layout": "out,in" if isinstance(module, torch.nn.Linear) else "in,out",
                         "shape": list(w.shape), "_input": [], "_output": []}
    def hook_for(name):
        def hook(module, args, output):
            for side, tensor in (("input", args[0]), ("output", output)):
                x = tensor.detach().double().reshape(-1, tensor.shape[-1])
                if not torch.isfinite(x).all():
                    raise ValueError("nonfinite activation")
                records[name]["_" + side].append((x.sum(0).cpu().numpy(),
                    (x*x).sum(0).cpu().numpy(), len(x)))
        return hook
    try:
        for name in tensors:
            hooks.append(modules[records[name]["module"]].register_forward_hook(hook_for(name)))
        metrics = ev.evaluate_model(model, sampled, [split])
    finally:
        for hook in hooks:
            hook.remove()
    for name, record in records.items():
        for side in ("input", "output"):
            chunks = record.pop("_" + side)
            count = sum(c[2] for c in chunks)
            if not count:
                raise ValueError("selected module was not called")
            mean = sum(c[0] for c in chunks) / count
            rms = np.sqrt(sum(c[1] for c in chunks) / count)
            record[side] = {"observations": count, "mean": mean.tolist(), "rms": rms.tolist(),
                            "mean_norm": float(np.linalg.norm(mean)),
                            "rms_norm": float(np.linalg.norm(rms)), "top_rms": _top(rms, top_k)}
        w = params[name].detach().double().cpu().numpy()
        row, col = np.linalg.norm(w, axis=1), np.linalg.norm(w, axis=0)
        record["weight"] = {"norm": float(np.linalg.norm(w)), "row_norms": row.tolist(),
                             "column_norms": col.tolist(), "top_rows": _top(row, top_k),
                             "top_columns": _top(col, top_k)}
    if core.sha256(directory / ev.MODEL_FILE) != before or core.sha256(directory / "config.json") != config_before:
        raise ValueError("source changed during diagnostics")
    return {**ev._provenance(root, model_dir, dataset), "split": split,
            "candidate_dir": candidate_dir, "device": device,
            "sampled_ids": [e["id"] for e in examples],
            "input_tokens": sum(len(e["tokens"]) for e in examples),
            "predicted_tokens": sum(len(e["tokens"])-1 for e in examples),
            "metrics": ev._strip(metrics), "tensors": records,
            "mode": "direct_inference_diagnostic",
            "interpretation": "Hypothesis evidence only; neither semantic labels nor causal proof. "
                              "Inference only: no gradients, backward, optimizer or persisted update.",
            "original_modified": False}
