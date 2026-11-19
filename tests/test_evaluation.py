import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from safetensors.numpy import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel

from weight_space_lab import core, evaluation as ev

TENSOR = "transformer.h.0.mlp.c_proj.weight"  # [hidden, embd]; axis 0 = hidden neurons


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _plan(root, strength=2 / 3, tensor=TENSOR, rows=None):
    n = core.inspect(root, "model/model.safetensors")
    shape = next(t["shape"] for t in n["tensors"] if t["name"] == tensor)
    return {"checkpoint": "model/model.safetensors", "source_sha256": n["sha256"],
            "hypothesis": "damaged neurons are over-amplified",
            "limits": {"max_changed_fraction": 1.0, "max_relative_l2": 1.0},
            "operations": [{"tensor": tensor, "kind": "scale", "axis": 0,
                            "indices": rows if rows is not None else list(range(shape[0])),
                            "strength": strength}]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Clean random model generates greedy data; the source is a damaged copy (x3)."""
    root = tmp_path_factory.mktemp("world")
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=32, n_positions=12, n_embd=16, n_layer=1, n_head=2,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    clean = GPT2LMHeadModel(config).eval()
    rng = np.random.default_rng(0)
    seqs = set()
    with torch.no_grad():
        for _ in range(1000):
            if len(seqs) >= 60:
                break
            ids = torch.tensor([rng.integers(32, size=2).tolist()])
            for _ in range(10):
                ids = torch.cat([ids, clean(input_ids=ids).logits[:, -1].argmax(-1, keepdim=True)], 1)
            seqs.add(tuple(ids[0].tolist()))
    assert len(seqs) == 60, "bounded fixture generation did not produce enough examples"
    seqs = sorted(seqs)
    split = {"discovery": seqs[:10], "validation": seqs[10:30], "test": seqs[30:]}
    data = {"name": "greedy-self", "version": "1", "splits": {
        k: [{"id": f"{k}{i}", "tokens": list(s)} for i, s in enumerate(v)] for k, v in split.items()}}
    (root / "data.json").write_text(json.dumps(data))
    with torch.no_grad():
        clean.state_dict()[TENSOR] *= 3.0
    clean.save_pretrained(root / "model", safe_serialization=True)
    return root


@pytest.fixture
def ws(world, tmp_path):
    shutil.copytree(world, tmp_path / "w")
    return tmp_path / "w"


def test_dataset_validation(ws):
    ds = ev.load_dataset(ws, "data.json")
    assert set(ds["splits"]) == {"discovery", "validation", "test"} and len(ds["sha256"]) == 64
    for bad in ({"splits": {}}, {"splits": {"a": []}}, {"splits": {"a": [{"id": "x", "tokens": [1]}]}},
                {"splits": {"a": [{"id": "x", "tokens": [1, True]}]}},
                {"splits": {"a": [{"id": "x", "tokens": [1, 2]}], "b": [{"id": "x", "tokens": [1, 2]}]}}):
        (ws / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            ev.load_dataset(ws, "bad.json")
    with pytest.raises(ValueError, match="inside workspace"):
        ev.load_dataset(ws, "../x.json")


def test_baseline_matches_manual_token_weighted_nll(ws):
    ds = ev.load_dataset(ws, "data.json")
    model = ev.load_causal_lm(ws, "model")
    got = ev.evaluate_model(model, ds, ["test"])["test"]
    total, count, hits = 0.0, 0, 0
    for ex in ds["splits"]["test"]:
        ids = torch.tensor([ex["tokens"]])
        with torch.no_grad():
            logits = model(input_ids=ids[:, :-1]).logits[0].double()
        lp = torch.log_softmax(logits, -1)
        total -= float(lp[torch.arange(11), ids[0, 1:]].sum())
        hits += int((logits.argmax(-1) == ids[0, 1:]).sum())
        count += 11
    assert got["tokens"] == count and got["nll"] == pytest.approx(total / count, rel=1e-6)
    assert got["accuracy"] == pytest.approx(hits / count)


def test_context_and_token_bounds(ws):
    model = ev.load_causal_lm(ws, "model")
    for tokens, msg in (([1] * 13, "context"), ([1, 99], "vocab")):
        ds = {"splits": {"s": [{"id": "e", "tokens": tokens}]}}
        with pytest.raises(ValueError, match=msg):
            ev.evaluate_model(model, ds)


def test_positive_candidate_passes_heldout_gate_and_preserves_original(ws):
    src = (ws / "model/model.safetensors").read_bytes()
    core.write_candidate(ws, _plan(ws), "cand")
    report = ev.evaluate_candidate(ws, "model", "cand", "data.json", min_nll_gain=0.01)
    assert (ws / "model/model.safetensors").read_bytes() == src
    assert report["gate"]["passed"], report["gate"]
    assert report["comparison"]["test"]["nll_gain"] > 0.01
    assert report["comparison"]["test"]["gain_ci95"][0] > 0
    for key in ("source_sha256", "config_sha256", "patch_sha256", "evaluator_sha256"):
        assert len(report[key]) == 64
    assert report["dataset"]["sha256"] and report["adoption"] == "recommended_not_applied"
    assert report["original_modified"] is False
    out = ev.write_report(ws, report, "reports/r.json")
    with pytest.raises(FileExistsError):
        ev.write_report(ws, report, out)


def test_identity_and_bad_gate_rejected(ws):
    core.write_candidate(ws, _plan(ws, strength=0.0), "ident")
    report = ev.evaluate_candidate(ws, "model", "ident", "data.json", min_nll_gain=0.001)
    assert report["edit"]["changed_parameters"] == 0 and not report["gate"]["passed"]
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="min_nll_gain"):
            ev.evaluate_candidate(ws, "model", "ident", "data.json", min_nll_gain=bad)
    with pytest.raises(ValueError, match="differ"):
        ev.evaluate_candidate(ws, "model", "ident", "data.json", min_nll_gain=.1, heldout_split="validation")


def test_worse_candidate_fails_gate(ws):
    # Destroy a separate attention projection, leaving the MLP damage intact.
    core.write_candidate(ws, _plan(ws, strength=1.0, tensor="transformer.h.0.attn.c_attn.weight"), "ablate")
    report = ev.evaluate_candidate(ws, "model", "ablate", "data.json", min_nll_gain=0.01)
    assert not report["gate"]["passed"] and report["adoption"] == "rejected"
    assert report["comparison"]["test"]["nll_gain"] < 0.01


def test_overlay_actually_changes_model(ws):
    core.write_candidate(ws, _plan(ws), "cand")
    model, info = ev.apply_candidate(ws, "model", "cand")
    patch = load_file(str(ws / "cand/replacements.safetensors"))[TENSOR]
    np.testing.assert_array_equal(model.state_dict()[TENSOR].numpy(), patch)
    assert info["changed_parameters"] == patch.size
    ids = torch.tensor([[1, 2, 3, 4]])
    baseline = ev.load_causal_lm(ws, "model")
    with torch.inference_mode():
        assert not torch.equal(model(input_ids=ids).logits, baseline(input_ids=ids).logits)


def test_invalid_evaluation_budgets_rejected(ws):
    core.write_candidate(ws, _plan(ws), "cand")
    for kwargs in ({"bootstrap": 0}, {"bootstrap": True}, {"seed": -1},
                   {"max_nll_regression": float("inf")}, {"max_accuracy_drop": -1}):
        with pytest.raises(ValueError):
            ev.evaluate_candidate(ws, "model", "cand", "data.json", min_nll_gain=.01, **kwargs)


def test_missing_parameter_checkpoint_rejected(ws):
    path = ws / "model/model.safetensors"
    tensors = load_file(str(path))
    tensors.pop(TENSOR)
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="incomplete"):
        ev.load_causal_lm(ws, "model")


def test_model_file_symlink_cannot_escape_workspace(ws, tmp_path):
    outside = tmp_path / "outside.safetensors"
    model_file = ws / "model/model.safetensors"
    model_file.rename(outside)
    model_file.symlink_to(outside)
    with pytest.raises(ValueError, match="inside workspace"):
        ev.load_causal_lm(ws, "model")


def _rewrite_manifest(ws, name, mutate_patch=None, mutate_manifest=None):
    d = ws / name
    patch = load_file(str(d / "replacements.safetensors"))
    if mutate_patch:
        patch = mutate_patch(patch)
        save_file(patch, str(d / "replacements.safetensors"))
    m = json.loads((d / "manifest.json").read_text())
    m["report"]["patch_sha256"] = core.sha256(d / "replacements.safetensors")
    if mutate_manifest:
        mutate_manifest(m)
    (d / "manifest.json").write_text(json.dumps(m))


def test_rejects_tampering(ws):
    core.write_candidate(ws, _plan(ws), "a")
    shutil.copytree(ws / "a", ws / "hashbad")
    p = load_file(str(ws / "hashbad/replacements.safetensors"))
    p[TENSOR][0, 0] += 1
    save_file(p, str(ws / "hashbad/replacements.safetensors"))
    with pytest.raises(ValueError, match="patch hash"):
        ev.verify_candidate(ws, "model", "hashbad")
    # consistent hash but values not derived from the plan
    shutil.copytree(ws / "a", ws / "vals")
    _rewrite_manifest(ws, "vals", lambda p: {k: v + 1 for k, v in p.items()})
    with pytest.raises(ValueError, match="differ from plan"):
        ev.verify_candidate(ws, "model", "vals")
    # injected unrelated tensor, hash fixed up
    shutil.copytree(ws / "a", ws / "inj")
    _rewrite_manifest(ws, "inj", lambda p: {**p, "transformer.wte.weight": np.zeros((32, 16), np.float32)})
    with pytest.raises(ValueError, match="keys differ"):
        ev.verify_candidate(ws, "model", "inj")
    # wrong dtype
    shutil.copytree(ws / "a", ws / "dt")
    _rewrite_manifest(ws, "dt", lambda p: {k: v.astype(np.float64) for k, v in p.items()})
    with pytest.raises(ValueError, match="dtype"):
        ev.verify_candidate(ws, "model", "dt")
    # extra file
    shutil.copytree(ws / "a", ws / "extra")
    (ws / "extra/evil.bin").write_text("x")
    with pytest.raises(ValueError, match="exactly"):
        ev.verify_candidate(ws, "model", "extra")
    # stale source
    shutil.copytree(ws / "a", ws / "stale")
    _rewrite_manifest(ws, "stale", mutate_manifest=lambda m: m["plan"].update(source_sha256="0" * 64))
    with pytest.raises(ValueError, match="source hash"):
        ev.verify_candidate(ws, "model", "stale")


def test_tied_weight_edit_rejected_not_silent(ws):
    # GPT-2 wte is tied with lm_head: the checkpoint stores only wte.
    plan = _plan(ws, strength=0.5, tensor="transformer.wte.weight", rows=[0, 1])
    core.write_candidate(ws, plan, "tied")
    with pytest.raises(ValueError, match="tied"):
        ev.apply_candidate(ws, "model", "tied")
    with pytest.raises(ValueError, match="tied"):
        ev.evaluate_candidate(ws, "model", "tied", "data.json", min_nll_gain=.01)


def test_unmapped_tensor_name_is_not_a_silent_noop(ws):
    # tensor exists in source file but not as a model parameter name -> reject
    path = ws / "model/model.safetensors"
    tensors = load_file(str(path))
    tensors["bogus.weight"] = np.ones((2, 2), np.float32)
    save_file(tensors, str(path))
    core.write_candidate(ws, _plan(ws, strength=0.5, tensor="bogus.weight", rows=[0]), "bogus")
    with pytest.raises(ValueError, match="not a model parameter"):
        ev.apply_candidate(ws, "model", "bogus")


def test_model_dir_must_be_local_and_inside_workspace(ws):
    with pytest.raises(ValueError, match="inside workspace"):
        ev.load_causal_lm(ws, "../model")
    with pytest.raises(ValueError, match="config.json"):
        ev.load_causal_lm(ws, "reports")
