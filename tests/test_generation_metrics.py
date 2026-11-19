import copy
import pytest
from weight_space_lab.benchmarks import summarize_generations, compare_generations


def outputs(tokens):
    return [{"eos_token_ids": [999], "examples": [{"id": "p", "continuation_ids": tokens,
                                                  "finish_reason": "max_new_tokens"}]}]


def settings():
    return {"min_repeat_gain": .05, "max_reference_nll_increase": .03,
            "max_reference_accuracy_drop": .02, "max_short_rate_increase": .05}


def test_repetition_eos_and_positive_control():
    repeated = summarize_generations(outputs(list(range(8)) * 16))
    unique = summarize_generations(outputs(list(range(128))))
    assert repeated["mean_repeat_fraction"] > .9 and unique["mean_repeat_fraction"] == 0
    eos = summarize_generations(outputs(list(range(32)) + [999] + [1] * 100))
    assert eos["mean_repeat_fraction"] == 0 and eos["per_prompt"][0]["ignored_post_eos_tokens"] == 100
    ref = {"nll": 2., "accuracy": .5}
    assert compare_generations(repeated, unique, ref, ref, settings())["decision"] == "gate_passed"


def test_short_identity_and_reference_regression_controls_rejected():
    repeated = summarize_generations(outputs(list(range(8)) * 16))
    empty = summarize_generations(outputs([]))
    unique = summarize_generations(outputs(list(range(128))))
    ref = {"nll": 2., "accuracy": .5}
    short = compare_generations(repeated, empty, ref, ref, settings())
    assert short["decision"] == "gate_failed" and not short["checks"]["short_rate_preserved"]
    assert compare_generations(repeated, repeated, ref, ref, settings())["decision"] == "gate_failed"
    regression = compare_generations(repeated, unique, ref, {"nll": 2.1, "accuracy": .5}, settings())
    assert regression["decision"] == "gate_failed" and not regression["checks"]["reference_nll_preserved"]


def test_prompt_pairing_and_missing_reference_rejected():
    a = summarize_generations(outputs(list(range(128))))
    b = copy.deepcopy(a); b["per_prompt"][0]["id"] = "different"
    with pytest.raises(ValueError, match="same prompts"):
        compare_generations(a, b, {}, {}, settings())
    with pytest.raises(ValueError, match="reference"):
        compare_generations(a, a, {}, {}, settings())


def test_single_prompt_concentration_guard_and_distributed_control():
    def batch(a, b):
        result = outputs(a)
        result[0]['examples'].append({'id': 'q', 'continuation_ids': b, 'finish_reason': 'max_new_tokens'})
        return summarize_generations(result)
    loop, unique = list(range(8))*16, list(range(128))
    ref = {'nll': 2., 'accuracy': .5}
    thresholds = {**settings(), 'leave_largest_prompt_out_gain_positive': True}
    concentrated = compare_generations(batch(loop, unique), batch(unique, unique), ref, ref, thresholds, require_ci=False)
    assert concentrated['checks']['mean_repeat_gain']
    assert not concentrated['checks']['largest_prompt_removed_gain_positive']
    assert concentrated['decision'] == 'gate_failed'
    distributed = compare_generations(batch(loop, loop), batch(unique, unique), ref, ref, thresholds)
    assert distributed['decision'] == 'gate_passed'
