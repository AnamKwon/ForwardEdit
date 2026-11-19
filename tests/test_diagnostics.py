import json
import numpy as np
import pytest
from weight_space_lab import core
from weight_space_lab.diagnostics import diagnose_model


def tiny_world(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=16, n_positions=16, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model.save_pretrained(tmp_path / 'model', safe_serialization=True)
    data = {'name': 'random-fixture', 'version': '1', 'splits': {
        'discovery': [{'id': 'd', 'tokens': [1, 2, 3, 4]}],
        'validation': [{'id': 'v', 'tokens': [2, 3, 4, 5]}],
        'test': [{'id': 't', 'tokens': [3, 4, 5, 6]}]}}
    (tmp_path / 'data.json').write_text(json.dumps(data))
    return model


def test_forward_evidence_without_gradients(tmp_path, monkeypatch):
    model = tiny_world(tmp_path)
    name = 'transformer.h.0.mlp.c_proj.weight'
    before = core.sha256(tmp_path / 'model/model.safetensors')
    report = diagnose_model(tmp_path, 'model', 'data.json', [name])
    tensor = report['tensors'][name]
    assert tensor['layout'] == 'in,out'
    assert tensor['input']['observations'] == 3
    assert len(tensor['input']['rms']) == 32
    assert len(tensor['output']['rms']) == 8
    # Independent actual hook for comparison, not a fabricated activation statistic.
    captured = []
    hook = model.transformer.h[0].mlp.c_proj.register_forward_hook(lambda m, a, o: captured.append(a[0].detach()))
    import torch
    model.eval()
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))
    hook.remove()
    expected = captured[0].double().square().mean((0, 1)).sqrt().numpy()
    np.testing.assert_allclose(tensor['input']['rms'], expected, atol=1e-9)
    assert 'gradient' not in tensor
    assert report['mode'] == 'direct_inference_diagnostic'
    assert report['predicted_tokens'] == 3
    assert core.sha256(tmp_path / 'model/model.safetensors') == before
    with pytest.raises(ValueError, match='disabled'):
        diagnose_model(tmp_path, 'model', 'data.json', [name], split='test', include_gradients=True)
    with pytest.raises(ValueError, match='tied'):
        diagnose_model(tmp_path, 'model', 'data.json', ['lm_head.weight'])
    with pytest.raises(ValueError, match='Linear or Conv1D'):
        diagnose_model(tmp_path, 'model', 'data.json', ['transformer.h.0.ln_1.weight'])


def test_linear_layout_and_bounds(tmp_path):
    tiny_world(tmp_path)
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16))
    model.save_pretrained(tmp_path / 'linear')
    name = 'model.layers.0.mlp.down_proj.weight'
    result = diagnose_model(tmp_path, 'linear', 'data.json', [name])
    assert result['tensors'][name]['layout'] == 'out,in'
    assert len(result['tensors'][name]['output']['mean']) == 8
    with pytest.raises(ValueError):
        diagnose_model(tmp_path, 'linear', 'data.json', [name], max_examples=33)
