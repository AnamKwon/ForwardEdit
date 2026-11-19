"""Pure CPU metrics for recorded forward outputs; never chooses model edits.

Repetition is a mechanical diagnostic, not a language-quality or ability score.
EOS content boundaries and trivial-short-output guards are explicit.
"""
import math
import numpy as np


def summarize_generations(outputs, ngram_size=8):
    if type(ngram_size) is not int or not 2 <= ngram_size <= 16:
        raise ValueError("ngram_size must be 2..16")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("provide recorded probe outputs")
    rows, seen = [], set()
    for output in outputs:
        eos_ids = set(output.get("eos_token_ids", []))
        for example in output["examples"]:
            eid, tokens = example["id"], example["continuation_ids"]
            if eid in seen:
                raise ValueError("each prompt id must be scored once")
            seen.add(eid)
            if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("continuation_ids must be nonnegative token integers")
            stop = next((i for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
            content = tokens[:stop]
            grams = [tuple(content[i:i + ngram_size]) for i in range(max(0, len(content) - ngram_size + 1))]
            repeated = len(grams) - len(set(grams))
            rows.append({"id": eid, "content_tokens": len(content), "ngrams": len(grams),
                "repeated_ngrams": repeated, "repeat_fraction": repeated / len(grams) if grams else 0.,
                "short": len(content) < 16, "finished_eos": stop < len(tokens),
                "ignored_post_eos_tokens": max(0, len(tokens) - stop - 1),
                "finish_reason": example.get("finish_reason", "unknown")})
    if not rows or len(rows) > 256:
        raise ValueError("score 1..256 prompts")
    return {"ngram_size": ngram_size, "prompts": len(rows), "per_prompt": rows,
        "mean_repeat_fraction": float(np.mean([r["repeat_fraction"] for r in rows])),
        "short_rate": float(np.mean([r["short"] for r in rows])),
        "median_content_tokens": float(np.median([r["content_tokens"] for r in rows])),
        "mean_content_tokens": float(np.mean([r["content_tokens"] for r in rows])),
        "eos_rate": float(np.mean([r["finished_eos"] for r in rows])),
        "cutoff_rate": float(np.mean([r["finish_reason"] == "max_new_tokens" for r in rows]))}


def compare_generations(baseline, candidate, baseline_reference, candidate_reference,
                        thresholds, require_ci=True, bootstrap=2000, seed=17):
    """Pair the same prompt IDs; require less repetition and reference/length preservation."""
    ids = [r["id"] for r in baseline["per_prompt"]]
    before = {r["id"]: r for r in baseline["per_prompt"]}
    after = {r["id"]: r for r in candidate["per_prompt"]}
    if set(before) != set(after) or baseline["ngram_size"] != candidate["ngram_size"]:
        raise ValueError("pair the same prompts and ngram definition")
    required = ("min_repeat_gain", "max_reference_nll_increase", "max_reference_accuracy_drop",
                "max_short_rate_increase")
    if any(type(thresholds.get(k)) not in (int, float) or not math.isfinite(thresholds[k])
           or thresholds[k] < 0 for k in required) or thresholds["min_repeat_gain"] <= 0:
        raise ValueError("finite nonnegative thresholds and positive minimum gain required")
    if type(bootstrap) is not int or not 1 <= bootstrap <= 10000:
        raise ValueError("bootstrap must be 1..10000")
    for reference in (baseline_reference, candidate_reference):
        if any(type(reference.get(k)) not in (int, float) or not math.isfinite(reference[k]) for k in ("nll", "accuracy")):
            raise ValueError("actual reference NLL/accuracy metrics are required")
    gains = np.array([before[i]["repeat_fraction"] - after[i]["repeat_fraction"] for i in ids])
    index = np.random.default_rng(seed).integers(0, len(gains), size=(bootstrap, len(gains)))
    ci = np.percentile(gains[index].mean(1), [2.5, 97.5]).tolist()
    nll_increase = candidate_reference["nll"] - baseline_reference["nll"]
    accuracy_change = candidate_reference["accuracy"] - baseline_reference["accuracy"]
    minimum_length = min(32., .75 * baseline["median_content_tokens"])
    checks = {"mean_repeat_gain": float(gains.mean()) >= thresholds["min_repeat_gain"],
        "gain_ci_lower_positive": ci[0] > 0 if require_ci else True,
        "reference_nll_preserved": nll_increase <= thresholds["max_reference_nll_increase"],
        "reference_accuracy_preserved": accuracy_change >= -thresholds["max_reference_accuracy_drop"],
        "short_rate_preserved": candidate["short_rate"] <= baseline["short_rate"] + thresholds["max_short_rate_increase"],
        "median_length_preserved": candidate["median_content_tokens"] >= minimum_length}
    gain_without_largest = float((gains.sum() - gains.max()) / (len(gains)-1)) if len(gains) > 1 else None
    if thresholds.get("leave_largest_prompt_out_gain_positive", False):
        checks["largest_prompt_removed_gain_positive"] = gain_without_largest is not None and gain_without_largest > 0
    regressions = [{"id": i, "repeat_increase": float(-gain)} for i, gain in zip(ids, gains) if gain < 0]
    return {"mean_repeat_gain": float(gains.mean()), "gain_ci95": ci,
        "gain_without_largest_prompt": gain_without_largest,
        "reference_nll_increase": nll_increase, "reference_accuracy_change": accuracy_change,
        "minimum_median_length": minimum_length, "regressed_prompts": regressions,
        "checks": checks, "decision": "gate_passed" if all(checks.values()) else "gate_failed",
        "interpretation": "Mechanical repetition gate only; not general language/math ability or narrative correctness."}
