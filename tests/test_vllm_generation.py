"""CPU provenance controls for the optional engine; never execute a model."""
import json
import numpy as np
import pytest
from safetensors.numpy import save_file,load_file
from weight_space_lab import core
from weight_space_lab.vllm_generation import materialize,seed_for


def test_exact_runtime_materialization_and_cache_binding(tmp_path):
    source=tmp_path/'model';source.mkdir()
    values=np.array([[.125,-.75],[1.,2.]],dtype=np.float32)
    save_file({'model.layers.0.mlp.down_proj.weight':values},source/'model.safetensors')
    (source/'config.json').write_text(json.dumps({'model_type':'qwen3_5_text'}))
    (source/'conditional_config.json').write_text(json.dumps({'model_type':'qwen3_5','text_config':{'eos_token_id':248044}}))
    c={'model_dir':'model','source_sha256':core.sha256(source/'model.safetensors')}
    target,candidate=materialize(tmp_path,c,{},'original')
    assert candidate is None
    assert np.array_equal(load_file(target/'model.safetensors')['model.language_model.layers.0.mlp.down_proj.weight'],values)
    assert json.loads((target/'config.json').read_text())['text_config']['eos_token_id']==248044
    assert core.sha256(source/'model.safetensors')==c['source_sha256']
    assert materialize(tmp_path,c,{},'original')[0]==target
    (source/'preprocessor_config.json').write_text('{}')
    assert materialize(tmp_path,c,{},'original')[0]!=target
    (target/'model.safetensors').chmod(0o644)
    with (target/'model.safetensors').open('r+b') as f:
        f.seek(-1,2);f.write(b'\x01')
    (source/'preprocessor_config.json').unlink()
    with pytest.raises(ValueError,match='binding mismatch'):
        materialize(tmp_path,c,{},'original')


def test_item_random_streams_are_stable_and_distinct():
    assert seed_for('a',0,4217)==seed_for('a',0,4217)
    assert len({seed_for(i,s,4217) for i in ['a','b'] for s in [0,1]})==4
