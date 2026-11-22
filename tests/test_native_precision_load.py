"""Compare effective loader values on explicit untrained Qwen fixtures."""
import json
import pytest
from weight_space_lab import research, research_worker as worker, core


@pytest.fixture
def qwen_source(tmp_path):
    import torch
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen3_5TextConfig'):
        pytest.skip('native Qwen3.5 loader integration requires newer Transformers')
    config = transformers.Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=['linear_attention', 'full_attention'],
        max_position_embeddings=64,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                         'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]},
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 256)
        # A value not representable in BF16 proves restoration from source bytes.
        model.model.layers[0].linear_attn.A_log.fill_(0.123456789)
    model.eval().requires_grad_(False)
    directory = tmp_path / 'model'
    model.save_pretrained(directory, safe_serialization=True)
    original_dtypes = {name: ('torch.float32' if name.endswith('A_log') else 'torch.bfloat16')
                       for name, _ in model.named_parameters()}
    (directory / 'source_dtypes.json').write_text(json.dumps(original_dtypes))
    return {'model_dir': 'model', 'device': 'cpu', 'inference_dtype': 'bfloat16'}


@pytest.mark.parametrize('edited', [False, True])
def test_native_load_matches_default_effective_parameters_and_buffers(tmp_path, qwen_source, edited):
    import torch
    c = qwen_source
    spec = {}
    if edited:
        tensor = 'model.layers.1.mlp.down_proj.weight'
        plan = {'checkpoint': 'model/model.safetensors',
                'source_sha256': core.sha256(tmp_path / 'model/model.safetensors'),
                'hypothesis': 'Explicit fixture value, not a fitted update',
                'limits': {'max_changed_fraction': 1, 'max_relative_l2': 1},
                'operations': [{'kind': 'set_values', 'tensor': tensor, 'rows': [0], 'columns': [0],
                                'expected_values': [[18/256]], 'values': [[0.1234567]]}]}
        # Read the exact fixture value; never derive a fitting target.
        from safetensors import safe_open
        with safe_open(tmp_path / 'model/model.safetensors', framework='pt') as source:
            plan['operations'][0]['expected_values'] = [[source.get_tensor(tensor)[0, 0].item()]]
        core.write_candidate(tmp_path, plan, 'candidate')
        spec['candidate_dir'] = 'candidate'
    which = 'edited' if edited else 'original'
    ordinary, changed = worker.load_verified(tmp_path, c, spec, which)
    native, actual = worker.load_verified(tmp_path, {**c, 'native_precision_load': True}, spec, which)
    assert actual == changed
    assert (actual > 0) == edited
    assert ordinary.state_dict().keys() == native.state_dict().keys()
    for name, before in ordinary.state_dict().items():
        after = native.state_dict()[name]
        assert before.dtype == after.dtype and torch.equal(before, after), name
    for (name, before), (other, after) in zip(ordinary.named_buffers(), native.named_buffers()):
        assert name == other and before.dtype == after.dtype and torch.equal(before, after), name
    sensitive = native.model.layers[0].linear_attn.A_log
    assert sensitive.dtype == torch.float32
    assert sensitive.flatten()[0].item() == torch.tensor(0.123456789).item()
    assert not native.training and all(not p.requires_grad for p in native.parameters())


def test_native_load_requires_dtype_provenance(tmp_path, qwen_source):
    (tmp_path / 'model/source_dtypes.json').unlink()
    with pytest.raises(ValueError, match='source_dtypes.json'):
        worker.load_verified(tmp_path, {**qwen_source, 'native_precision_load': True}, {}, 'original')


@pytest.mark.parametrize('value', ['yes', 1, None])
def test_native_precision_configuration_requires_boolean(tmp_path, value):
    c = dict(successful_forward_budget=1, attempt_budget=1, max_items=1,
             max_new_tokens=1, max_input_tokens=2, max_total_input_tokens=2,
             device='cpu', inference_dtype='bfloat16', native_precision_load=value)
    (tmp_path / 'research.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='must be boolean'):
        research.config(tmp_path)
