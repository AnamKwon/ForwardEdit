import copy
import numpy as np
import pytest
from safetensors.numpy import save_file
from weight_space_lab.core import prepare, sha256, write_candidate


def plan(tmp_path, axis=0):
    w = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    save_file({'w': w}, str(tmp_path / 'm.safetensors'))
    return w, {'checkpoint': 'm.safetensors', 'source_sha256': sha256(tmp_path / 'm.safetensors'),
        'hypothesis': 'manual additive correction', 'limits': {'max_changed_fraction': 1, 'max_relative_l2': 1},
        'operations': [{'tensor': 'w', 'kind': 'rank_one', 'axis': axis, 'indices': [2, 0],
            'strength': .5, 'left': [1, -2] if axis == 0 else [1, -2, 3],
            'right': [2, 0, -1, 1] if axis == 0 else [2, -1]}]}


@pytest.mark.parametrize('axis', [0, 1])
def test_manual_matrix(tmp_path, axis):
    w, p = plan(tmp_path, axis)
    actual, report = prepare(tmp_path, p)
    delta = np.array([[1, 0, -.5, .5], [-2, 0, 1, -1]]) if axis == 0 else np.array([[1, -.5], [-2, 1], [3, -1.5]])
    expected = w.copy()
    if axis == 0:
        expected[[2, 0], :] += delta
    else:
        expected[:, [2, 0]] += delta
    np.testing.assert_array_equal(actual['w'], expected)
    assert report['changed_parameters'] == np.count_nonzero(expected-w)
    write_candidate(tmp_path, p, 'candidate')
    assert sha256(tmp_path / 'm.safetensors') == p['source_sha256']


@pytest.mark.parametrize('field,value', [('left', [1]), ('right', [float('nan')]*4),
    ('strength', -0.1), ('indices', [3]), ('axis', True)])
def test_invalid(tmp_path, field, value):
    _, p = plan(tmp_path)
    p['operations'][0][field] = value
    with pytest.raises(ValueError):
        prepare(tmp_path, p)


def test_identity_and_budgets(tmp_path):
    w, p = plan(tmp_path)
    for key in ('max_changed_fraction', 'max_relative_l2'):
        bad = copy.deepcopy(p)
        bad['limits'][key] = 1e-8
        with pytest.raises(ValueError, match='budget'):
            prepare(tmp_path, bad)
    p['operations'][0]['left'] = [0, 0]
    np.testing.assert_array_equal(prepare(tmp_path, p)[0]['w'], w)
    p['source_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        prepare(tmp_path, p)
