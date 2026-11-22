"""Read-only comparison of matched, verified research generations."""
from collections import Counter
from pathlib import Path

from . import core, evaluation, research
from .research_analysis import _job


def _load_generation(root, run):
    if not isinstance(run, dict) or set(run) != {"job_id", "model"}:
        raise ValueError("a generation reference requires job_id and model")
    model = run["model"]
    if model not in ("original", "edited"):
        raise ValueError("model must be original or edited")
    _, snapshot, artifacts = _job(root, run["job_id"])
    spec = snapshot["spec"]
    if spec["type"] != "generate":
        raise ValueError("generation job required")
    selected = [a for a in artifacts if a["kind"] == "generations" and a["model"] == model]
    if len(selected) != 1:
        raise ValueError("one verified artifact for the requested model required")
    artifact = selected[0]
    path = core.local(root, artifact["path"])
    if core.sha256(path) != artifact["sha256"]:
        raise ValueError("generation artifact hash mismatch")
    rows = research.read(root, path)
    keys = [(row["id"], row["sample"]) for row in rows]
    if keys != [(key, sample) for key in spec["ids"] for sample in range(spec["n"])]:
        raise ValueError("generation rows do not follow the recorded batch order")
    source = {
        **run,
        "generation_sha256": artifact["sha256"],
        "source_sha256": snapshot["config"]["source_sha256"],
        "candidate_dir": spec.get("candidate_dir") if model == "edited" else None,
        "max_new_tokens": spec["max_new_tokens"],
        "first_token_logit_bias": spec.get("first_token_logit_bias", []),
    }
    if source["candidate_dir"]:
        patch = str(Path(source["candidate_dir"]) / "replacements.safetensors")
        source["candidate_patch_sha256"] = snapshot["hashes"][patch]
    return source, snapshot, rows


def _outcome(row, ngram_size, envelope):
    tokens = row["continuation_ids"]
    if len(tokens) != row["generated_tokens"] or len(row["input_ids"]) != row["input_tokens"]:
        raise ValueError("generation token counts differ from saved token IDs")
    tail = tokens[-2048:]
    counts = Counter(tuple(tail[i:i + ngram_size]) for i in range(len(tail) - ngram_size + 1))
    answer = row["extracted_answer"]
    stopped = row["finish_reason"] == "eos"
    terminal_envelope = None
    if envelope is not None:
        lines = row["text"].strip().splitlines()
        line = lines[-1].strip() if lines else ""
        start, end = envelope
        terminal_envelope = (
            answer is not None and len(line) >= len(start) + len(end)
            and line.startswith(start) and line.endswith(end)
            and line[len(start):-len(end)].strip() == str(answer).strip()
        )
    return {
        "grade": row["grade"],
        "extracted_answer": answer,
        "finish_reason": row["finish_reason"],
        "generated_tokens": len(tokens),
        "terminal_answer_envelope": terminal_envelope,
        "completed_correct": row["grade"] is True and stopped and terminal_envelope is not False,
        "tail_tokens_checked": len(tail),
        "peak_tail_ngram_occurrences": max(counts.values(), default=0),
        "repeated_tail_ngram_positions": sum(count for count in counts.values() if count > 1),
    }


def research_compare_generations(root, reference, candidate, output, ids=None,
                                 ngram_size=16, final_answer_envelope=None):
    """Locate behavioral changes on matched inputs without a model call or a new grade."""
    root = Path(root)
    research.integer(ngram_size, 1, 64, "ngram_size")
    if final_answer_envelope is not None and (
        not isinstance(final_answer_envelope, list) or len(final_answer_envelope) != 2
        or any(not isinstance(value, str) or not 1 <= len(value) <= 128 or "\n" in value or "\r" in value
               for value in final_answer_envelope)
    ):
        raise ValueError("final_answer_envelope must be two nonempty literal single-line delimiters")
    original, ref_snapshot, ref_rows = _load_generation(root, reference)
    edited, edit_snapshot, edit_rows = _load_generation(root, candidate)
    left_spec, right_spec = ref_snapshot["spec"], edit_snapshot["spec"]
    for key in ("split", "ids", "n"):
        if left_spec[key] != right_spec[key]:
            raise ValueError("generation split, samples or batch order differ")
    # A workspace binds its model and execution configuration to the ledger.
    # Compare explicitly as well to keep the matched condition in the report.
    if ref_snapshot["config"] != edit_snapshot["config"]:
        raise ValueError("generation configurations differ")
    default_data = ref_snapshot["config"].get("dataset_path")
    reference_data = left_spec.get("dataset_path", default_data)
    candidate_data = right_spec.get("dataset_path", default_data)
    if reference_data != candidate_data:
        raise ValueError("generation datasets differ; compare scores under the same stored labels")
    for left, right in zip(ref_rows, edit_rows):
        if left["input_ids"] != right["input_ids"]:
            raise ValueError("generation prompt token IDs differ")
    if ids is not None and (
        not isinstance(ids, list) or not ids or any(not isinstance(key, str) for key in ids)
        or len(set(ids)) != len(ids) or not set(ids) <= set(left_spec["ids"])
    ):
        raise ValueError("ids must select unique recorded examples")
    selected = set(left_spec["ids"] if ids is None else ids)
    results = []
    for left, right in zip(ref_rows, edit_rows):
        if left["id"] not in selected:
            continue
        a, b = left["continuation_ids"], right["continuation_ids"]
        common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        divergence = None
        if len(a) != common or len(b) != common:
            divergence = {
                "continuation_offset": common,
                "matched_prefix_tokens": len(left["input_ids"]) + common,
                "reference_next_token_id": a[common] if common < len(a) else None,
                "candidate_next_token_id": b[common] if common < len(b) else None,
            }
        results.append({
            "id": left["id"], "sample": left["sample"],
            "first_divergence": divergence, "token_delta": len(b) - len(a),
            "reference": _outcome(left, ngram_size, final_answer_envelope),
            "candidate": _outcome(right, ngram_size, final_answer_envelope),
        })
    counts = {"samples": len(results), "diverged": sum(row["first_divergence"] is not None for row in results)}
    for side in ("reference", "candidate"):
        counts[side] = {
            "correct": sum(row[side]["grade"] is True for row in results),
            "completed_correct": sum(row[side]["completed_correct"] for row in results),
            "cutoffs": sum(row[side]["finish_reason"] == "max_new_tokens" for row in results),
            "generated_tokens": sum(row[side]["generated_tokens"] for row in results),
        }
    counts["completed_repairs"] = sum(
        not row["reference"]["completed_correct"] and row["candidate"]["completed_correct"] for row in results
    )
    counts["completed_regressions"] = sum(
        row["reference"]["completed_correct"] and not row["candidate"]["completed_correct"] for row in results
    )
    config = ref_snapshot["config"]
    report = {
        "reference": original, "candidate": edited, "split": left_spec["split"],
        "generation_backend": config.get("generation_backend", "transformers"),
        "inference_dtype": config["inference_dtype"], "batch_size": config.get("batch_size", 8),
        "sampling": config["sampling"], "ngram_size": ngram_size,
        "final_answer_envelope": final_answer_envelope,
        "interpretation": "Token divergence and exact repeated windows locate observations. They do not identify semantic errors, causes or repair coordinates.",
        "summary": counts, "rows": results,
    }
    path = evaluation.write_report(root, report, output)
    return {"report_path": path, "sha256": core.sha256(core.local(root, path)), "summary": counts,
            "model_forward_calls": 0}
