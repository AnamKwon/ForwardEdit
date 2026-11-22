"""Data leakage preflight: no model, weights, training or forward needed."""
import json

import pytest

from weight_space_lab import research


def load(root, discovery, validation):
    (root / 'data.json').write_text(json.dumps({'splits': {
        'discovery': discovery, 'validation': validation,
    }}))
    return research.dataset(root, 'data.json')


@pytest.mark.parametrize('left,right', [
    ({'prompt': 'same question', 'answer': 'yes'},
     {'prompt': 'same question', 'answer': 'no'}),
    ({'prompt': 'question\r\nnext'}, {'prompt': ' question\nnext\n'}),
    ({'tokens': [1, 2], 'prompt': 'old wording'},
     {'tokens': [1, 2], 'prompt': 'new wording'}),
    ({'tokens': [1, 2], 'prompt': 'same wording'},
     {'tokens': [3, 4], 'prompt': 'same wording'}),
    ({'reference': 'same teacher sequence'}, {'reference': 'same teacher sequence'}),
])
def test_renaming_or_relabeling_input_does_not_create_validation(tmp_path, left, right):
    with pytest.raises(ValueError, match='cross-split input overlap'):
        load(tmp_path, [{'id': 'train', **left}], [{'id': 'validation', **right}])


def test_grouped_variants_stay_together(tmp_path):
    train = [{'id': 'train', 'prompt': 'one variant', 'group_id': 'source-17'}]
    validation = [{'id': 'validation', 'prompt': 'another variant', 'group_id': 'source-17'}]
    with pytest.raises(ValueError, match='cross-split group overlap'):
        load(tmp_path, train, validation)
    validation[0]['group_id'] = 'source-18'
    assert load(tmp_path, train, validation)['validation'] == validation


@pytest.mark.parametrize('group', ['', '   ', 7, None, 'a' * 257])
def test_invalid_group_identity_is_not_silently_ignored(tmp_path, group):
    with pytest.raises(ValueError, match='group_id'):
        load(tmp_path, [{'id': 'train', 'prompt': 'question', 'group_id': group}], [])


def test_image_bytes_distinguish_inputs_and_aliases_do_not_hide_overlap(tmp_path):
    # The loader checks identity, not image decoding. No Pillow/model dependency.
    (tmp_path / 'one.png').write_bytes(b'first image')
    (tmp_path / 'alias.png').write_bytes(b'first image')
    (tmp_path / 'two.png').write_bytes(b'second image')
    train = [{'id': 'train', 'prompt': 'What is visible?', 'images': ['one.png']}]
    validation = [{'id': 'validation', 'prompt': 'What is visible?', 'images': ['alias.png']}]
    with pytest.raises(ValueError, match='cross-split input overlap'):
        load(tmp_path, train, validation)
    validation[0]['images'] = ['two.png']
    assert load(tmp_path, train, validation)['validation'] == validation


def test_repeated_observations_within_editing_split_remain_usable(tmp_path):
    train = [{'id': 'first', 'prompt': 'same', 'group_id': 'case'},
             {'id': 'second', 'prompt': 'same', 'group_id': 'case'}]
    validation = [{'id': 'validation', 'prompt': 'unseen', 'group_id': 'other'}]
    assert load(tmp_path, train, validation)['discovery'] == train


def test_leakage_is_rejected_before_a_forward_job_can_be_reserved(tmp_path):
    with pytest.raises(ValueError, match='cross-split input overlap'):
        load(tmp_path, [{'id': 'a', 'tokens': [1, 2]}], [{'id': 'b', 'tokens': [1, 2]}])
    request = {'type': 'generate', 'model': 'original', 'split': 'discovery',
               'ids': ['a'], 'max_new_tokens': 1}
    # validate_spec reads data before model loading or ledger reservation.
    with pytest.raises(ValueError, match='cross-split input overlap'):
        research.validate_spec(tmp_path, {'dataset_path': 'data.json', 'max_items': 2}, request)
    assert not (tmp_path / 'research_jobs').exists()
