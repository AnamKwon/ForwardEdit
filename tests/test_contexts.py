import json
import pytest
from weight_space_lab.contexts import build_repetition_contexts
from weight_space_lab.evaluation import load_dataset


def make_inputs(root, continuation=None, **extra):
    ref = [41, 42] + list(range(100, 150))
    (root / 'refs.json').write_text(json.dumps({'splits': {'discovery': [{'id': 'x', 'tokens': ref}],
        'test': [{'id': 'hidden', 'tokens': [1, 3, 5]}]}}))
    probe = {'source_sha256': 'a'*64, 'candidate_dir': None, 'split': 'discovery', 'eos_token_ids': [99],
             'examples': [{'id': 'x', 'input_ids': ref[:2], 'continuation_ids': list(range(8))*4 if continuation is None else continuation}], **extra}
    (root / 'probe.json').write_text(json.dumps(probe))
    return probe


def test_real_loop_matches_exact_position_and_roundtrips(tmp_path):
    make_inputs(tmp_path)
    result = build_repetition_contexts(tmp_path, 'refs.json', ['probe.json'], ['x'], 'contexts.json')
    pair = result['pairs'][0]
    assert pair['reference_positions'] == pair['failure_positions'] == [24, 25, 26, 27]
    assert all(w['generated_occurrences'] >= 3 and w['reference_occurrences'] == 1 for w in pair['witnesses'])
    assert len(load_dataset(tmp_path, 'contexts.json')['splits']['discovery']) == 2
    assert result['model_forward_calls'] == result['parameter_updates'] == 0


@pytest.mark.parametrize('tokens', [list(range(8))*2, [1, 2, 1, 2, 1, 2], [99]+list(range(8))*4])
def test_no_fake_loop_or_eos_tail(tmp_path, tokens):
    make_inputs(tmp_path, tokens)
    with pytest.raises(ValueError, match='no actual recurring'):
        build_repetition_contexts(tmp_path, 'refs.json', ['probe.json'], ['x'], 'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_holdout_prompt_mismatch_and_missing_ids(tmp_path):
    probe = make_inputs(tmp_path)
    with pytest.raises(ValueError, match='discovery reference'):
        build_repetition_contexts(tmp_path, 'refs.json', ['probe.json'], ['hidden'], 'bad.json')
    probe['examples'][0]['input_ids'] = [88, 89]
    (tmp_path/'probe.json').write_text(json.dumps(probe))
    with pytest.raises(ValueError, match='prompt mismatch'):
        build_repetition_contexts(tmp_path, 'refs.json', ['probe.json'], ['x'], 'bad.json')
    make_inputs(tmp_path, split='test')
    with pytest.raises(ValueError, match='discovery probes'):
        build_repetition_contexts(tmp_path, 'refs.json', ['probe.json'], ['x'], 'bad.json')
