"""Inspectable structure, source and bounded actual forward observations.

Tools expose calculations; they never select an edit or certify semantic roles.
Observed call order is not a complete computation graph. Identity edges only mean
that the same live tensor object was observed at two selected module boundaries.
"""
import hashlib
import inspect as pyinspect
import math
from pathlib import Path
import weakref

import numpy as np
from safetensors import safe_open
from . import core, evaluation as ev
from .inference import load_inference_model


def read_parameter_entries(root, checkpoint, tensor, flat_indices):
    """Read explicit coordinates of scalars/vectors/matrices/higher-rank tensors."""
    root = Path(root)
    path = core.local(root, checkpoint)
    before = core.sha256(path)
    with safe_open(path, framework="np") as handle:
        shape = handle.get_slice(tensor).get_shape()
        core._indices(flat_indices, math.prod(shape))
        if len(flat_indices) > 4096:
            raise ValueError("read <=4096 entries")
        entries = []
        for index in flat_indices:
            coordinates = [int(i) for i in np.unravel_index(index, shape)] if shape else []
            # Slice only the requested entry, including for very large embeddings.
            value = (handle.get_slice(tensor)[tuple(slice(i, i + 1) for i in coordinates)]
                     if shape else handle.get_tensor(tensor))
            dtype = str(value.dtype)
            entries.append({"flat_index": index, "coordinates": coordinates, "value": value.reshape(-1)[0].item()})
    if core.sha256(path) != before:
        raise ValueError("checkpoint changed while reading")
    return {"source_sha256": before, "tensor": tensor, "shape": shape,
            "dtype": dtype, "entries": entries, "model_forward_calls": 0}


def describe_model_structure(root, model_dir, prefix="", max_modules=128, offset=0):
    if (type(max_modules) is not int or not 1 <= max_modules <= 256
            or type(offset) is not int or offset < 0 or not isinstance(prefix, str)):
        raise ValueError("max_modules1..256, nonnegative offset and string prefix required")
    model = load_inference_model(Path(root), model_dir)
    modules = [(n, m) for n, m in model.named_modules() if n == prefix or n.startswith(prefix + ".") or not prefix]
    if prefix and not modules:
        raise ValueError("unknown module prefix")
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    rows = []
    for name, module in modules[offset:offset + max_modules]:
        parameters = [{"name": (name + "." if name else "") + leaf,
                       "shape": list(p.shape), "dtype": str(p.dtype),
                       "aliases": aliases[id(p)]} for leaf, p in module.named_parameters(recurse=False)]
        rows.append({"name": name, "parent": name.rpartition(".")[0] if name else None,
                     "class": type(module).__module__ + "." + type(module).__qualname__,
                     "children": [n for n, _ in module.named_children()], "parameters": parameters})
    directory = ev._model_dir(Path(root), model_dir)
    return {"source_sha256": core.sha256(directory / ev.MODEL_FILE),
            "config_sha256": core.sha256(directory / "config.json"),
            "model_type": getattr(model.config, "model_type", None), "modules": rows,
            "total_matching_modules": len(modules), "next_offset": offset + max_modules if offset + max_modules < len(modules) else None,
            "unique_parameters": sum(p.numel() for p in model.parameters()),
            "model_forward_calls": 0,
            "interpretation": "Ownership hierarchy and shared parameter identities only; module order is not execution order or a causal graph."}


def inspect_module_computation(root, model_dir, modules, candidate_dir=None):
    if (not isinstance(modules, list) or not 1 <= len(modules) <= 12
            or any(not isinstance(n, str) for n in modules) or len(set(modules)) != len(modules)):
        raise ValueError("select 1..12 unique module names")
    torch, _ = ev._torch()
    from transformers.pytorch_utils import Conv1D
    model = load_inference_model(Path(root), model_dir, candidate_dir)
    named = dict(model.named_modules())
    rows = []
    for name in modules:
        if name not in named:
            raise ValueError("unknown module: " + name)
        module = named[name]
        equation = None
        if type(module) is torch.nn.Linear:
            equation = "y = x @ W.T + bias; W[out,in]"
        elif type(module) is Conv1D:
            equation = "y = x @ W + bias; W[in,out]"
        elif type(module) is torch.nn.Embedding:
            equation = "y = W[token_index]; W[index,feature]"
        elif type(module) is torch.nn.LayerNorm:
            equation = "y = (x-mean(x))/sqrt(var(x,unbiased=False)+eps) * weight + bias over normalized_shape"
        try:
            source = pyinspect.getsource(pyinspect.unwrap(module.forward))
            filename = pyinspect.getsourcefile(pyinspect.unwrap(module.forward))
        except (OSError, TypeError):
            source, filename = None, None
        rows.append({"name": name, "class": type(module).__module__ + "." + type(module).__qualname__,
            "equation": equation, "equation_status": "standard_module_equation" if equation else "read_actual_source",
            "parameters": {leaf: {"shape": list(p.shape), "dtype": str(p.dtype)}
                           for leaf, p in module.named_parameters(recurse=False)},
            "forward_source": None if source is None else source[:8000], "source_file": filename,
            "forward_source_sha256": None if source is None else hashlib.sha256(source.encode()).hexdigest(),
            "source_truncated": source is not None and len(source) > 8000})
    return {"modules": rows, "candidate_dir": candidate_dir, "model_forward_calls": 0,
            "interpretation": "Actual loaded module forward source and standard arithmetic, not inferred neuron semantics. Functional operations may live in parent methods/helpers."}


def analyze_matrix_geometry(root, checkpoint, tensor, rows, columns, rank=4,
                            right_operand=None):
    """SVD of an explicit block; optional explicit block product A @ B (weight algebra only)."""
    if type(rank) is not int or not 1 <= rank <= 8:
        raise ValueError("rank must be1..8")
    if max(len(rows), len(columns)) > 64:
        raise ValueError("select <=64 rows/columns")
    path = core.local(Path(root), checkpoint)
    before = core.sha256(path)
    a = core.read_matrix(Path(root), checkpoint, tensor, rows, columns)
    matrix = np.asarray(a["values"], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("finite weights required")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(rank, len(s))
    result = {"source_sha256": before, "tensor": tensor, "rows": rows, "columns": columns,
              "singular_values": s.tolist(), "left_vectors": u[:, :k].tolist(), "right_vectors": vt[:k].tolist(),
              "row_norms": np.linalg.norm(matrix, axis=1).tolist(),
              "column_norms": np.linalg.norm(matrix, axis=0).tolist(), "product": None}
    if right_operand is not None:
        if not isinstance(right_operand, dict):
            raise ValueError("right_operand needs tensor,rows,columns")
        if max(len(right_operand["rows"]), len(right_operand["columns"])) > 64:
            raise ValueError("right operand <=64 rows/columns")
        b = core.read_matrix(Path(root), checkpoint, right_operand["tensor"], right_operand["rows"], right_operand["columns"])
        other = np.asarray(b["values"], dtype=np.float64)
        if not np.isfinite(other).all() or matrix.shape[1] != other.shape[0]:
            raise ValueError("same source, finite compatible A @ B blocks required")
        result["product"] = (matrix @ other).tolist()
    if core.sha256(path) != before:
        raise ValueError("checkpoint changed during geometry analysis")
    result.update(model_forward_calls=0, parameter_updates=0,
        interpretation="Weight geometry/product only. Composition omits intermediate nonlinearities, norms, gates, attention and residual paths; not equivalent to full network execution or a semantic/circuit proof.")
    return result


def _first_tensor(value, torch, path):
    if torch.is_tensor(value):
        return value, path
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else []
    for key, child in children:
        found = _first_tensor(child, torch, path + "[" + str(key) + "]")
        if found is not None:
            return found
    return None


def trace_token_flow(root, model_dir, dataset_path, selections, examples, output,
                     split="discovery", candidate_dir=None, device="cpu"):
    """Observe <=16 selected boundaries on <=4 explicit examples/8 positions each."""
    if split not in ("discovery", "validation"):
        raise ValueError("flow tracing permits development only")
    if not isinstance(selections, list) or not 1 <= len(selections) <= 16:
        raise ValueError("select1..16 modules")
    names = [s["module"] for s in selections]
    if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("unique module names required")
    for spec in selections:
        for key in ("input_features", "output_features"):
            ids = spec[key]
            if not isinstance(ids, list) or len(ids) > 512 or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("explicit unique0..512 feature indices per side required")
    if not isinstance(examples, list) or not 1 <= len(examples) <= 4:
        raise ValueError("select1..4 explicit examples")
    root = Path(root)
    ds = ev.load_dataset(root, dataset_path)
    available = {e["id"]: e for e in ds["splits"].get(split, [])}
    seen = set()
    for example in examples:
        eid, positions = example["id"], example["positions"]
        if eid not in available or eid in seen:
            raise ValueError("unique existing development ids required")
        seen.add(eid)
        if not isinstance(positions, list) or not 1 <= len(positions) <= 8:
            raise ValueError("select1..8 prediction positions")
        core._indices(positions, len(available[eid]["tokens"]) - 1)
    if any(len(available[eid]["tokens"]) > 256 for eid in seen):
        raise ValueError("flow trace <=256 tokens/example")
    cells = sum(len(s[k]) for s in selections for k in ('input_features', 'output_features')) * sum(len(e['positions']) for e in examples)
    if cells > 65536:
        raise ValueError("flow trace selected scalar budget<=65536")
    model = load_inference_model(root, model_dir, candidate_dir, device)
    torch, _ = ev._torch()
    named = dict(model.named_modules())
    if any(name not in named for name in names):
        raise ValueError("unknown selected module")
    directory = ev._model_dir(root, model_dir)
    before = core.sha256(directory / ev.MODEL_FILE)
    config_hash = core.sha256(directory / "config.json")
    patch_path = None if candidate_dir is None else core.local(root, candidate_dir) / "replacements.safetensors"
    patch_hash = None if patch_path is None else core.sha256(patch_path)
    handles, records = [], []
    events, producers, pending, positions = [], {}, {}, []

    def summary(found, indices):
        nonlocal cells_recorded
        if found is None:
            return None
        tensor, path = found
        info = {"tensor_path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "samples": [],
                "sampling_layout_assumption": "batch,token,last_axis for rank3; check actual module source"}
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            if indices:
                core._indices(indices, tensor.shape[-1])
            for p in positions:
                if p >= tensor.shape[1]:
                    info["samples"].append({"position": p, "status": "axis1_has_no_such_position"})
                    continue
                vector = tensor[0, p].detach().double()
                cells_recorded += len(indices)
                if cells_recorded > 65536:
                    raise ValueError("flow trace selected scalar budget<=65536 including repeated module calls")
                if not torch.isfinite(vector).all():
                    raise ValueError("nonfinite flow activation")
                info["samples"].append({"position": p, "features": indices,
                    "signed_values": vector[indices].cpu().tolist(), "full_feature_l2": float(vector.norm().cpu())})
        return info

    def pre(spec):
        def observe(module, args, kwargs):
            if len(events) >= 64:
                raise ValueError("selected modules exceeded64 calls/example")
            found = _first_tensor(args, torch, "args") or _first_tensor(kwargs, torch, "kwargs")
            previous = None if found is None else producers.get(id(found[0]))
            source_event = previous[1] if previous is not None and previous[0]() is found[0] else None
            event = {"event": len(events), "module": spec["module"],
                     "input": summary(found, spec["input_features"]),
                     "input_same_tensor_as_event_output": source_event,
                     "input_link_status": "observed_live_tensor_identity" if source_event is not None else "unobserved_or_functional_path"}
            events.append(event)
            pending.setdefault(spec["module"], []).append(event)
        return observe

    def post(spec):
        def observe(module, args, result):
            event = pending[spec["module"]].pop()
            found = _first_tensor(result, torch, "output")
            event["output"] = summary(found, spec["output_features"])
            if found is not None:
                producers[id(found[0])] = (weakref.ref(found[0]), event["event"])
        return observe

    cells_recorded = 0
    try:
        for spec in selections:
            handles.append(named[spec["module"]].register_forward_pre_hook(pre(spec), with_kwargs=True))
            handles.append(named[spec["module"]].register_forward_hook(post(spec)))
        with torch.inference_mode():
            for example in examples:
                tokens, positions = available[example["id"]]["tokens"], example["positions"]
                if len(tokens) > ev._context_limit(model) or max(tokens) >= model.config.vocab_size:
                    raise ValueError("input exceeds actual context/vocabulary")
                events, producers, pending = [], {}, {}
                result = model(input_ids=torch.tensor([tokens[:-1]], device=device))
                predictions = []
                for p in positions:
                    logp = result.logits[0, p].double().log_softmax(-1)
                    if not torch.isfinite(logp).all():
                        raise ValueError("nonfinite output")
                    predictions.append({"position": p, "input_token": tokens[p], "target_token": tokens[p + 1],
                        "target_log_probability": float(logp[tokens[p + 1]].cpu()), "predicted_token": int(logp.argmax().cpu())})
                records.append({"id": example["id"], "events": events, "predictions": predictions,
                                "selected_modules_not_called": sorted(set(names) - {e["module"] for e in events}),
                                "prefix_sha256": ev._canon_hash(tokens[:-1])})
    finally:
        for handle in handles:
            handle.remove()
    if (core.sha256(directory / ev.MODEL_FILE) != before or core.sha256(directory / "config.json") != config_hash
            or core.sha256(core.local(root, dataset_path)) != ds["sha256"]
            or (patch_path is not None and core.sha256(patch_path) != patch_hash)):
        raise ValueError("flow inputs changed")
    report = {"kind": "observed_token_flow", "source_sha256": before, "config_sha256": config_hash,
              "dataset_sha256": ds["sha256"], "candidate_dir": candidate_dir, "patch_sha256": patch_hash,
              "split": split, "device": device, "examples": records, "gradients_enabled": False,
              "optimizer_steps": 0,
              "interpretation": "Actual selected boundary observations only. Tensor identity edges are partial; execution order is not causality. Functional residuals, attention products, gates and unselected modules can hide connections. Read actual forward source; semantic roles require controlled interventions."}
    ev.write_report(root, report, output)
    return {"report_path": output, "report_sha256": core.sha256(core.local(root, output)), "report": report}
