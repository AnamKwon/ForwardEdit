"""Parameter meaning, readable forward probes and development feedback only.

No autograd, optimizer, numerical gradients or automatic fitting. CUDA is used
only to run inference; checkpoint editing remains explicit CPU array arithmetic.
"""
from pathlib import Path

from . import core, evaluation as ev


def load_inference_model(root, model_dir, candidate_dir=None, device="cpu"):
    torch, _ = ev._torch()
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    model = (ev.load_causal_lm(root, model_dir) if candidate_dir is None
             else ev.apply_candidate(root, model_dir, candidate_dir)[0])
    return model.to(device).eval().requires_grad_(False)


def describe_parameter(root, model_dir, tensor, rows, columns, candidate_dir=None):
    """Map actual module/layout to a bounded slice of actual current parameter values."""
    torch, _ = ev._torch()
    from transformers.pytorch_utils import Conv1D
    model = load_inference_model(root, model_dir, candidate_dir)
    owner, _, leaf = tensor.rpartition(".")
    module = dict(model.named_modules()).get(owner)
    state = model.state_dict()
    if leaf != "weight" or not isinstance(module, (torch.nn.Linear, Conv1D)):
        raise ValueError("parameter must be a Linear or Conv1D weight")
    if tensor in ev._tied_names(model):
        raise ValueError("tied/shared weight is not a supported edit target")
    w = state[tensor]
    core._indices(rows, w.shape[0]); core._indices(columns, w.shape[1])
    if len(rows) * len(columns) > 4096:
        raise ValueError("read at most 4096 values")
    linear = isinstance(module, torch.nn.Linear)
    role = "matrix projection; semantic feature role requires inference/intervention evidence"
    if ".mlp." in tensor:
        if any(part in tensor for part in ("down_proj", "c_proj")):
            role = "MLP hidden features -> residual stream; each input-feature slice writes a residual direction"
        elif any(part in tensor for part in ("up_proj", "c_fc")):
            role = "residual stream -> MLP hidden features"
        elif "gate_proj" in tensor:
            role = "MLP multiplicative gate preactivation"
    elif any(part in tensor for part in ("attn", "attention")):
        role = "attention projection; query/key/value/head interpretation depends on the concrete architecture"
    return {"tensor": tensor, "owner": owner, "module_class": type(module).__name__,
        "shape": list(w.shape), "layout": "out,in" if linear else "in,out",
        "forward_equation": "y = x @ W.T + bias" if linear else "y = x @ W + bias",
        "input_feature_axis": 1 if linear else 0, "output_feature_axis": 0 if linear else 1,
        "structural_role": role, "semantic_role_status": "hypothesis_required",
        "rows": rows, "columns": columns, "values": w[rows][:, columns].cpu().tolist(),
        "source_sha256": core.sha256(ev._model_dir(root, model_dir) / ev.MODEL_FILE),
        "candidate_dir": candidate_dir, "requires_grad": False,
        "update_rule": "LLM supplies explicit new values; no gradients or optimizer"}


def score_candidate(root, model_dir, candidate_dir, dataset_path, splits=None, device="cpu"):
    """Feedback on explicitly requested development splits; never score heldout during editing."""
    splits = ["discovery", "validation"] if splits is None else splits
    if (not isinstance(splits, list) or not splits or len(set(splits)) != len(splits)
            or any(s not in ("discovery", "validation") for s in splits)):
        raise ValueError("editing feedback permits discovery/validation only")
    ds = ev.load_dataset(Path(root), dataset_path)
    for split in splits:
        if split not in ds["splits"]:
            raise ValueError("unknown split")
        if len(ds["splits"][split]) > 256 or sum(len(e["tokens"]) for e in ds["splits"][split]) > 32768:
            raise ValueError("development inference budget exceeded")
    before = core.sha256(ev._model_dir(root, model_dir) / ev.MODEL_FILE)
    model = load_inference_model(root, model_dir, candidate_dir, device)
    metrics = ev.evaluate_model(model, ds, splits)
    if core.sha256(ev._model_dir(root, model_dir) / ev.MODEL_FILE) != before:
        raise ValueError("source changed during inference")
    return {**ev._provenance(root, model_dir, ds), "candidate_dir": candidate_dir,
        "device": device, "metrics": metrics, "mode": "inference_only_development_feedback",
        "final_acceptance": False, "gradients_enabled": False, "optimizer_steps": 0}


def probe_model(root, model_dir, dataset_path, candidate_dir=None, split="discovery",
                max_examples=2, top_k=5, max_new_tokens=16, device="cpu", stop_at_eos=False,
                use_cache=False, _confirmation_path=None):
    """Actual next-token top-k and greedy continuations; bounded CPU/CUDA inference."""
    if (type(max_examples) is not int or not 1 <= max_examples <= 8
            or type(top_k) is not int or not 1 <= top_k <= 20
            or type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 256
            or type(stop_at_eos) is not bool or type(use_cache) is not bool):
        raise ValueError("probe permits 1..8 examples, top_k1..20, 0..256 new tokens, boolean stop_at_eos")
    confirmation = None
    if split not in ("discovery", "validation"):
        if split != "test" or _confirmation_path is None:
            raise ValueError("editing probes permit discovery/validation only; heldout requires frozen confirmation")
        from .confirmation import validate_confirmation
        confirmation = validate_confirmation(root, _confirmation_path, model_dir, dataset_path, candidate_dir,
            max_new_tokens, stop_at_eos, use_cache, device)
    torch, transformers = ev._torch()
    ds = ev.load_dataset(Path(root), dataset_path)
    if split not in ds["splits"]:
        raise ValueError("unknown split")
    source = core.sha256(ev._model_dir(root, model_dir) / ev.MODEL_FILE)
    model = load_inference_model(root, model_dir, candidate_dir, device)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    eos_ids = eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
    eos_ids = {i for i in eos_ids if type(i) is int and 0 <= i < model.config.vocab_size}
    examples = ds["splits"][split][:max_examples]
    if sum(len(e["tokens"]) for e in examples) > 2048:
        raise ValueError("probe input token budget exceeded (2048)")
    tokenizer = None
    directory = ev._model_dir(root, model_dir)
    if (directory / "tokenizer_config.json").is_file():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False)
    output = []
    with torch.inference_mode():
        for example in examples:
            tokens = example["tokens"]
            if len(tokens) + max_new_tokens > ev._context_limit(model) or max(tokens) >= model.config.vocab_size:
                raise ValueError("probe exceeds context or vocabulary")
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            initial = model(ids, use_cache=True) if use_cache else model(ids)
            logp = initial.logits[:, -1].float().log_softmax(-1)
            if not torch.isfinite(logp).all():
                raise ValueError("nonfinite inference output")
            values, indices = logp.topk(min(top_k, logp.shape[-1]))
            generated = ids.clone()
            cached_logits = initial.logits[:, -1]
            past = getattr(initial, "past_key_values", None) if use_cache else None
            finish_reason = "max_new_tokens" if max_new_tokens else "no_generation"
            for step in range(max_new_tokens):
                logits = cached_logits if step == 0 or past is not None else model(generated).logits[:, -1]
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite generation output")
                next_token = logits.argmax(-1, keepdim=True)
                generated = torch.cat([generated, next_token], -1)
                if stop_at_eos and int(next_token.item()) in eos_ids:
                    finish_reason = "eos"
                    break
                if past is not None and step + 1 < max_new_tokens:
                    decoded = model(next_token, past_key_values=past, use_cache=True)
                    cached_logits, past = decoded.logits[:, -1], decoded.past_key_values
            continuation = generated[0, len(tokens):].cpu().tolist()
            row = {"id": example["id"], "input_ids": tokens,
                "next_token_topk": [{"token_id": int(i), "log_probability": float(v)}
                                    for i, v in zip(indices[0].cpu(), values[0].cpu())],
                "continuation_ids": continuation, "generated_tokens": len(continuation),
                "finish_reason": finish_reason}
            if tokenizer is not None:
                row["input_text"] = tokenizer.decode(tokens)
                row["continuation_text"] = tokenizer.decode(continuation)
            output.append(row)
    if core.sha256(ev._model_dir(root, model_dir) / ev.MODEL_FILE) != source:
        raise ValueError("source changed during probe")
    return {"source_sha256": source, "candidate_dir": candidate_dir, "split": split,
        "device": device, "decoding": "greedy; stop at EOS or limit" if stop_at_eos else "greedy; fixed count, no early stop",
        "stop_at_eos": stop_at_eos, "eos_token_ids": sorted(eos_ids), "examples": output,
        "use_cache": use_cache,
        "phase": "heldout_confirmation" if confirmation else "development", "confirmation": confirmation,
        "mode": "inference_only", "gradients_enabled": False, "optimizer_steps": 0}
