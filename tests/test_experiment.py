import json
import pytest
from weight_space_lab import core, evaluation as ev
from weight_space_lab.experiment import run_experiment
from test_diagnostics import tiny_world


def config(tmp_path, strength=.01):
    return {'model_dir': 'model', 'dataset_path': 'data.json', 'min_nll_gain': 1e-12,
        'bootstrap': 10, 'max_accuracy_drop': 1, 'plans': [{
            'checkpoint': 'model/model.safetensors',
            'source_sha256': core.sha256(tmp_path / 'model/model.safetensors'),
            'hypothesis': 'small local intervention',
            'limits': {'max_changed_fraction': 1, 'max_relative_l2': 1},
            'operations': [{'tensor': 'transformer.h.0.mlp.c_proj.weight', 'axis': 0,
                'indices': [0], 'strength': strength, 'kind': 'scale'}]}]}


def test_no_winner_does_not_touch_heldout(tmp_path, monkeypatch):
    tiny_world(tmp_path)
    calls = []
    original = ev.evaluate_model
    def observed(model, dataset, splits=None, **kw):
        calls.append(splits)
        assert 'test' not in splits
        return original(model, dataset, splits, **kw)
    monkeypatch.setattr(ev, 'evaluate_model', observed)
    monkeypatch.setattr(ev, 'evaluate_candidate', lambda *a, **k: pytest.fail('no winner confirmation'))
    result = run_experiment(tmp_path, config(tmp_path, 0), 'run')
    assert result['status'] == 'no_winner' and result['confirmation_calls'] == 0
    assert len(calls) == 3
    assert not (tmp_path / 'run/confirmation.json').exists()
    with pytest.raises(FileExistsError):
        run_experiment(tmp_path, config(tmp_path, 0), 'run')


def test_freeze_then_single_confirmation(tmp_path, monkeypatch):
    tiny_world(tmp_path)
    original = ev.evaluate_model
    calls = []
    def selection(model, dataset, splits=None, **kw):
        assert splits == ['validation']
        metrics = original(model, dataset, splits, **kw)
        calls.append(splits)
        if len(calls) >= 3:
            metrics['validation']['nll'] -= .1
        return metrics
    monkeypatch.setattr(ev, 'evaluate_model', selection)
    confirmations = []
    def confirmation(*args, **kwargs):
        frozen = json.loads((tmp_path / 'run/frozen-choice.json').read_text())
        assert frozen['winner']['index'] == 0
        ev.verify_candidate(tmp_path, 'model', frozen['winner']['candidate_dir'])
        confirmations.append(args[2])
        return {'behavior_validation': 'gate_failed'}
    monkeypatch.setattr(ev, 'evaluate_candidate', confirmation)
    result = run_experiment(tmp_path, config(tmp_path), 'run')
    assert result['winner'] == 0 and len(confirmations) == 1
    assert len(calls) == 3
    assert result['status'] == 'gate_failed'
    assert result['rejected_candidates'] == [0]


def test_config_rejections(tmp_path):
    tiny_world(tmp_path)
    for field, value in [('plans', []), ('selection_split', 'test'), ('preserve_splits', ['test']),
                         ('min_nll_gain', 0), ('bootstrap', 0), ('max_examples', 0)]:
        c = config(tmp_path)
        c[field] = value
        with pytest.raises(ValueError):
            run_experiment(tmp_path, c, 'unused')
    assert not (tmp_path / 'unused').exists()


def test_gradient_based_design_is_rejected(tmp_path):
    tiny_world(tmp_path)
    c = config(tmp_path)
    c['design_mode'] = 'gradient_assisted'
    with pytest.raises(ValueError, match='disabled'):
        run_experiment(tmp_path, c, 'forbidden')
    assert not (tmp_path / 'forbidden').exists()


def test_preserve_regression_rejects_selection_winner(tmp_path, monkeypatch):
    tiny_world(tmp_path)
    original = ev.evaluate_model
    count = 0
    def observed(model, dataset, splits=None, **kw):
        nonlocal count
        assert splits == ['validation', 'discovery']
        result = original(model, dataset, splits, **kw)
        count += 1
        if count == 3:
            result['validation']['nll'] -= .1
            result['discovery']['nll'] += .1
        return result
    monkeypatch.setattr(ev, 'evaluate_model', observed)
    monkeypatch.setattr(ev, 'evaluate_candidate', lambda *a, **kw: pytest.fail('regressed winner'))
    c = config(tmp_path)
    c['preserve_splits'] = ['discovery']
    result = run_experiment(tmp_path, c, 'preserve')
    assert result['status'] == 'no_winner'
    evidence = json.loads((tmp_path / 'preserve/selection-00.json').read_text())
    assert evidence['comparison']['validation']['nll_gain'] > c['min_nll_gain']
    assert evidence['selection_passed'] is False
