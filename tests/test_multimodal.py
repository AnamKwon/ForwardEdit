import json

import pytest

from weight_space_lab import multimodal, research_worker


def test_named_input_observation_records_pre_normalization_values():
    torch = pytest.importorskip('torch')
    import numpy as np
    model = torch.nn.Module()
    model.norm = torch.nn.LayerNorm(3)
    captured = []

    class Artifacts:
        def save(self, value, kind, which, **metadata):
            captured.append((value, kind, metadata))

    obs = research_worker.Observations(model, {'module_inputs': ['norm']}, Artifacts(), 'original')
    obs.batch, obs.phase = [{'id': 'case', 'tokens': [1]}], 'prompt'
    with torch.inference_mode():
        model.norm(torch.tensor([[[1., 2., 3.]]]))
    obs.flush()
    obs.close()
    np.testing.assert_array_equal(captured[0][0], [[1., 2., 3.]])
    assert captured[0][1] == 'module_inputs'
    assert captured[0][2]['module'] == 'norm'


def test_image_pixels_and_replay_reach_batch(tmp_path):
    torch = pytest.importorskip('torch')
    image_module = pytest.importorskip('PIL.Image')
    image_module.new('RGB', (2, 2), (255, 0, 0)).save(tmp_path / 'red.png')

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]['content'][0]['type'] == 'image'
            return messages[0]['content'][1]['text']

        def __call__(self, images, text, **kwargs):
            assert text == 'Which color?'
            assert images[0].getpixel((0, 0)) == (255, 0, 0)
            return {'input_ids': torch.tensor([[4, 5, 6]]),
                    'attention_mask': torch.ones((1, 3)),
                    'token_type_ids': torch.tensor([[0, 1, 0]]),
                    'pixel_values': torch.tensor([[[[1.]], [[0.]], [[0.]]]])}

    row = {'id': 'image', 'images': ['red.png'], 'prompt': 'Which color?', 'tokens': [4, 5, 6, 7]}
    prepared = research_worker.tokenize(None, [row], 'generate', '', root=tmp_path, processor=Processor())[0]
    batch = multimodal.batch_image_inputs([prepared, {'tokens': [1, 2]}], 4, 'cpu', torch.float32)
    assert prepared['tokens'] == [4, 5, 6, 7]
    assert batch['token_type_ids'].tolist() == [[0, 1, 0, 0], [0, 0, 0, 0]]
    assert batch['pixel_values'].flatten().tolist() == [1., 0., 0.]
    with pytest.raises(ValueError, match='prompt prefix'):
        multimodal.image_row(tmp_path, Processor(), {**row, 'tokens': [4, 0, 6]}, '')


def test_native_gemma_vision_executes_once_during_cached_generation(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from packaging.version import Version
    if Version(transformers.__version__) < Version('5.22'):
        pytest.skip('native Gemma3Model attachment requires Transformers 5.22')
    from safetensors.torch import save_file
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3ForCausalLM
    from transformers import Gemma3TextConfig, SiglipVisionConfig

    text_config = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, sliding_window=16, eos_token_id=2, pad_token_id=0,
        layer_types=['full_attention'])
    vision_config = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=2)
    config = Gemma3Config(text_config=text_config, vision_config=vision_config,
        mm_tokens_per_image=4, image_token_index=31, boi_token_index=29, eoi_token_index=30,
        eos_token_id=2, pad_token_id=0)
    original = Gemma3ForConditionalGeneration(config)
    text = Gemma3ForCausalLM(text_config)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.fill_(0.05)
        for parameter in text.parameters():
            parameter.fill_(0.05)
    state = {name.removeprefix('model.'): value.detach().contiguous()
             for name, value in original.state_dict().items()
             if name.startswith(('model.vision_tower.', 'model.multi_modal_projector.'))}
    state = {name.replace('vision_tower.', 'vision_tower.vision_model.', 1): value
             for name, value in state.items()}
    save_file(state, tmp_path / 'vision.safetensors')
    (tmp_path / 'config.json').write_text(config.to_json_string())
    settings = {'kind': 'gemma3', 'vision_checkpoint': 'vision.safetensors',
                'config': 'config.json', 'processor_dir': '.'}
    full = multimodal.attach_gemma3(tmp_path, text, settings, 'cpu', torch.float32,
                                  research_worker.cast_for_inference)
    assert full.model.language_model is text.model
    assert full.lm_head is text.lm_head
    assert all(not parameter.requires_grad for parameter in full.parameters())
    observed = []
    handle = full.model.vision_tower.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs['pixel_values'].clone()), with_kwargs=True)

    class Tokenizer:
        pad_token_id = 0
        def decode(self, ids, **kwargs):
            return ' '.join(map(str, ids))

    pixels = torch.ones((1, 3, 8, 8))
    rows = [{'id': 'image', 'tokens': [1, 31, 31, 31, 31, 3], 'answer': '',
             '_image_inputs': {'pixel_values': pixels, 'token_type_ids': [0, 1, 1, 1, 1, 0]}}]
    spec = {'n': 1, 'max_new_tokens': 2}
    config = {'batch_size': 1, 'sampling': {'seed': 1, 'temperature': 1., 'top_k': 1,
                                          'top_p': 1., 'presence_penalty': 0.}}
    obs = research_worker.Observations(full, {}, None, 'original')
    outputs, metrics, calls = research_worker.generate(full, Tokenizer(), rows, config, spec, obs)
    handle.remove()
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], pixels)
    assert outputs[0]['input_ids'] == rows[0]['tokens']
    assert metrics['samples'] == 1 and calls >= 1
