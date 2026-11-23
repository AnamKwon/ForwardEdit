"""Native frozen Gemma vision attachment and local image preprocessing."""
import json
import hashlib
from pathlib import Path

from . import core


def validate_settings(root, settings):
    if (not isinstance(settings, dict) or settings.get('kind') != 'gemma3'
            or set(settings) != {'kind', 'vision_checkpoint', 'config', 'processor_dir'}):
        raise ValueError('multimodal requires gemma3, vision_checkpoint, config and processor_dir')
    for name in ('vision_checkpoint', 'config'):
        if not core.local(Path(root), settings[name]).is_file():
            raise ValueError('missing multimodal ' + name)
    if not core.local(Path(root), settings['processor_dir']).is_dir():
        raise ValueError('missing multimodal processor directory')


def load_processor(root, settings):
    from transformers import Gemma3Processor
    validate_settings(root, settings)
    return Gemma3Processor.from_pretrained(
        str(core.local(Path(root), settings['processor_dir'])),
        local_files_only=True, trust_remote_code=False,
    )


def attach_gemma3(root, text_model, settings, device, dtype, cast):
    import torch
    from safetensors.torch import load_file
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    validate_settings(root, settings)
    config = Gemma3Config.from_dict(json.loads(core.local(Path(root), settings['config']).read_text()))
    config.text_config = text_model.config
    if text_model.config.model_type != 'gemma3_text':
        raise ValueError('Gemma vision attachment requires a Gemma3 text checkpoint')
    with torch.device('meta'):
        model = Gemma3ForConditionalGeneration(config)
    if not hasattr(model, 'model') or not hasattr(model.model, 'language_model'):
        raise ValueError('native Gemma3Model layout requires Transformers 5.22 or later')
    checkpoint = core.local(Path(root), settings['vision_checkpoint'])
    state = load_file(str(checkpoint), device='cpu')
    prefixes = ('vision_tower.', 'multi_modal_projector.')
    if not state or any(not name.startswith(prefixes) for name in state):
        raise ValueError('vision checkpoint contains unrelated or missing components')
    components = {}
    for name in ('vision_tower', 'multi_modal_projector'):
        previous = getattr(model.model, name)
        component_config = config.vision_config if name == 'vision_tower' else config
        component = type(previous)(component_config)
        prefix = name + '.'
        values = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        key_mapping = {key: key for key in values}
        if set(values) != set(component.state_dict()):
            from transformers.conversion_mapping import get_model_conversion_mapping
            from transformers.core_model_loading import WeightRenaming, rename_source_key
            conversions = get_model_conversion_mapping(component)
            if any(not isinstance(conversion, WeightRenaming) for conversion in conversions):
                raise ValueError('vision checkpoint requires unsupported numerical tensor conversion')
            key_mapping = {key: rename_source_key(key, conversions, [])[0] for key in values}
            if len(set(key_mapping.values())) != len(values):
                raise ValueError('vision checkpoint key conversion has duplicate targets')
            values = {key_mapping[key]: value for key, value in values.items()}
        component.load_state_dict(values, strict=True)
        source_dtypes = {key: str(value.dtype) for key, value in values.items()}
        components[name] = cast(component, device, dtype, source_dtypes)
        components[name]._research_precision_policy.update(
            checkpoint_key_renamings=sum(key != value for key, value in key_mapping.items()),
            checkpoint_key_mapping_sha256=hashlib.sha256(
                json.dumps(key_mapping, sort_keys=True).encode()).hexdigest(),
        )
    model.model.vision_tower = components['vision_tower']
    model.model.multi_modal_projector = components['multi_modal_projector']
    model.model.language_model = text_model.model
    model.lm_head = text_model.lm_head
    if any(value.is_meta for value in model.parameters()):
        raise ValueError('multimodal assembly left unmaterialized parameters')
    if model.model.language_model is not text_model.model or model.lm_head is not text_model.lm_head:
        raise ValueError('multimodal assembly changed verified text components')
    # Keep the original source-module names for observation and edit provenance.
    object.__setattr__(model, '_research_observation_model', text_model)
    model._research_precision_policy = {
        **getattr(text_model, '_research_precision_policy', {}),
        'multimodal_kind': 'gemma3', 'vision_checkpoint_sha256': core.sha256(checkpoint),
        'text_components_shared_by_identity': True,
        'vision_precision': components['vision_tower']._research_precision_policy,
        'projector_precision': components['multi_modal_projector']._research_precision_policy,
    }
    return model.eval().requires_grad_(False)


def image_row(root, processor, row, prompt_suffix):
    from PIL import Image

    images = []
    try:
        for name in row['images']:
            with Image.open(core.local(Path(root), name)) as source:
                images.append(source.convert('RGB'))
        content = [{'type': 'image'} for _ in images]
        content.append({'type': 'text', 'text': row['prompt'] + prompt_suffix})
        text = processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True,
        )
        values = processor(images=images, text=text, return_tensors='pt')
    finally:
        for image in images:
            image.close()
    allowed = {'input_ids', 'attention_mask', 'pixel_values', 'token_type_ids'}
    if set(values) - allowed:
        raise ValueError('unsupported Gemma processor output fields: ' + str(set(values) - allowed))
    ids = values['input_ids'][0].tolist()
    types = values['token_type_ids'][0].tolist()
    if 'tokens' in row:
        supplied = row['tokens']
        if supplied[:len(ids)] != ids:
            raise ValueError('image replay tokens differ from the processor prompt prefix')
        types += [0] * (len(supplied) - len(ids))
        ids = supplied
    return {**row, 'tokens': ids, '_image_inputs': {
        'pixel_values': values['pixel_values'], 'token_type_ids': types,
    }}


def batch_image_inputs(chunk, width, device, dtype):
    import torch

    images = [row['_image_inputs']['pixel_values'] for row in chunk if '_image_inputs' in row]
    if not images:
        return {}
    types = []
    for row in chunk:
        positions = row.get('_image_inputs', {}).get('token_type_ids', [0] * len(row['tokens']))
        types.append([0] * (width - len(positions)) + positions)
    return {
        'pixel_values': torch.cat(images, dim=0).to(device=device, dtype=dtype),
        'token_type_ids': torch.tensor(types, device=device, dtype=torch.long),
    }
