import json

import pytest

from weight_space_lab import core, research_analysis, trajectory


def logged_row(identifier, tokens, text, grade, finish, sample=0):
    return {
        "id": identifier, "sample": sample, "input_ids": [1, 2, 3], "input_tokens": 3,
        "continuation_ids": tokens, "generated_tokens": len(tokens), "text": text,
        "grade": grade, "extracted_answer": "blue", "finish_reason": finish,
    }


def archive(root, rows, job, model, ids, n=1):
    path = root / f"{job}.json"
    path.write_text(json.dumps(rows))
    spec = {"type": "generate", "split": "discovery", "ids": ids, "n": n, "max_new_tokens": 16}
    if model == "edited":
        spec["candidate_dir"] = "candidates/edit"
    cfg = {"source_sha256": "same", "inference_dtype": "float32", "sampling": {"seed": 4}}
    snapshot = {"spec": spec, "config": cfg, "hashes": {"candidates/edit/replacements.safetensors": "edit hash"}}
    return {}, snapshot, [{"path": path.name, "kind": "generations", "model": model, "sha256": core.sha256(path)}]


def test_divergence_cutoff_and_envelope_are_separate(tmp_path, monkeypatch):
    original = [
        logged_row("a", [4, 5, 2], "<answer>blue</answer>", True, "eos"),
        logged_row("b", [4, 5], "<answer>blue</answer>", True, "eos"),
    ]
    edited = [
        logged_row("a", [4, 9, 9, 9, 9], 'Instruction says "<answer>blue</answer>"\nStill thinking.', True, "max_new_tokens"),
        logged_row("b", [4, 5, 7], "First clause.\n<answer>blue</answer>\n", True, "eos"),
    ]
    jobs = {
        "j1": archive(tmp_path, original, "j1", "original", ["a", "b"]),
        "j2": archive(tmp_path, edited, "j2", "edited", ["a", "b"]),
    }
    jobs["j1"][1]["spec"]["first_token_logit_bias"] = [{"token_id": 4, "bias": .5}]
    monkeypatch.setattr(trajectory, "_job", lambda root, job: jobs[job])
    refs = ({"job_id": "j1", "model": "original"}, {"job_id": "j2", "model": "edited"})
    result = trajectory.research_compare_generations(tmp_path, *refs, "report.json", ngram_size=2,
                                                     final_answer_envelope=["<answer>", "</answer>"])
    assert result["model_forward_calls"] == 0
    assert result["summary"]["candidate"] == {"correct": 2, "completed_correct": 1, "cutoffs": 1, "generated_tokens": 8}
    assert result["summary"]["completed_regressions"] == 1
    report = json.loads((tmp_path / "report.json").read_text())
    first, second = report["rows"]
    assert first["first_divergence"] == {
        "continuation_offset": 1, "matched_prefix_tokens": 4,
        "reference_next_token_id": 5, "candidate_next_token_id": 9,
    }
    assert first["candidate"]["peak_tail_ngram_occurrences"] == 3
    assert first["candidate"]["terminal_answer_envelope"] is False
    assert second["first_divergence"]["reference_next_token_id"] is None
    assert report["candidate"]["candidate_patch_sha256"] == "edit hash"
    assert report["reference"]["first_token_logit_bias"] == [{"token_id": 4, "bias": .5}]
    assert report["candidate"]["first_token_logit_bias"] == []
    with pytest.raises(FileExistsError):
        trajectory.research_compare_generations(tmp_path, *refs, "report.json")

    edited[0]["input_ids"][0] = 42
    jobs["j2"] = archive(tmp_path, edited, "j2", "edited", ["a", "b"])
    with pytest.raises(ValueError, match="prompt token IDs differ"):
        trajectory.research_compare_generations(tmp_path, *refs, "mismatch.json", ids=["b"])
    (tmp_path / "j2.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        trajectory.research_compare_generations(tmp_path, *refs, "tampered.json")


def test_exact_replay_retains_sample_and_full_prompt(tmp_path, monkeypatch):
    rows = [logged_row("a", [4, 5], "", None, "eos"), logged_row("a", [6, 7], "", None, "eos", sample=1)]
    job = archive(tmp_path, rows, "j1", "original", ["a"], n=2)
    monkeypatch.setattr(research_analysis, "_job", lambda *args: job)
    report = research_analysis.research_make_contexts(
        tmp_path, "j1", ["a"], "context.json", sample=1, continuation_lengths={"a": 1},
    )
    saved = json.loads((tmp_path / report["dataset_path"]).read_text())
    assert saved["splits"]["discovery"] == [{"id": "a", "tokens": [1, 2, 3, 6]}]
    assert saved["provenance"]["sample"] == 1
    assert saved["provenance"]["continuation_lengths"] == {"a": 1}
    with pytest.raises(ValueError, match="exact prefix exceeds"):
        research_analysis.research_make_contexts(
            tmp_path, "j1", ["a"], "clipped.json", max_tokens=3, continuation_lengths={"a": 1},
        )
