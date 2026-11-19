"""The public evidence checker detects corruption without running experiments."""
import hashlib
import json

import pytest

from scripts.verify_public_evidence import verify_bundle


def bundle(root, data=b'{"gain": -0.1, "accepted": false}\n'):
    file = root / 'case/result.json'
    file.parent.mkdir()
    file.write_bytes(data)
    record = {'path': 'case/result.json', 'source': 'runs/case/result.json',
              'source_sha256': '0' * 64,  # Original and redacted bytes may differ.
              'published_sha256': hashlib.sha256(data).hexdigest(), 'published_bytes': len(data)}
    manifest = {'schema_version': 1, 'kind': 'curated_read_only_experiment_evidence', 'files': [record]}
    (root / 'MANIFEST.json').write_text(json.dumps(manifest))
    return file, manifest


def test_redacted_copy_verifies_and_tampering_fails(tmp_path):
    file, _ = bundle(tmp_path)
    assert verify_bundle(tmp_path)['files'] == 1
    file.write_text('{"gain": 0.1, "accepted": true}\n')
    with pytest.raises(ValueError, match='published bytes changed'):
        verify_bundle(tmp_path)


@pytest.mark.parametrize('path', ['../outside.json', '/outside.json', 'case/../result.json'])
def test_paths_cannot_escape_the_bundle(tmp_path, path):
    _, manifest = bundle(tmp_path)
    manifest['files'][0]['path'] = path
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe or duplicate'):
        verify_bundle(tmp_path)


def test_matching_hash_does_not_make_invalid_json_valid(tmp_path):
    bundle(tmp_path, b'{"gain": NaN}\n')
    with pytest.raises(ValueError, match='non-finite JSON'):
        verify_bundle(tmp_path)


def test_unlisted_assets_are_rejected(tmp_path):
    bundle(tmp_path)
    (tmp_path / 'checkpoint.safetensors').write_bytes(b'unexpected asset')
    with pytest.raises(ValueError, match='unlisted evidence files'):
        verify_bundle(tmp_path)
