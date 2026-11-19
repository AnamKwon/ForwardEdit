"""Frozen inference backend. Executed only by the research subprocess entry point."""
from __future__ import annotations
import argparse
import os
from fractions import Fraction
import hashlib
import json
import inspect
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import numpy as np
from safetensors import safe_open
from . import core, evaluation as ev, inference
from . import research as api


def tripwires():
    from .inference_guards import install_inference_guards
    install_inference_guards()


def boxed(text):
    """Last balanced box; a trailing unclosed box invalidates extraction."""
    starts = list(re.finditer(r'\\boxed\s*\{', text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def numeric(text):
    if text is None or len(text) > 1024:
        return None
    text = text.strip()
    if ',' in text:
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', text):
            return None
        text = text.replace(',', '')
    text = re.sub(r'\\(?:dfrac|tfrac)', r'\\frac', text)
    m = re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', text)
    if m:
        text = ('-' if m[1] == '-' else '') + m[2] + '/' + m[3]
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', text):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def grade(text, answer):
    extracted = boxed(text)
    ref = str(answer)
    if '\\boxed' in ref:
        ref = boxed(ref)
    elif '####' in ref:
        ref = ref.rsplit('####', 1)[1].strip()
    value, expected = numeric(extracted), numeric(ref)
    if value is not None and expected is not None:
        return extracted, value == expected
    if not extracted or not ref or len(extracted) > 1024 or len(ref) > 1024:
        return extracted, None
    # A bare identifier alone does not establish a mathematical answer. Keep
    # the existing ambiguous-symbol behavior even when the parser is installed.
    if re.fullmatch(r'[A-Za-z]+', extracted.strip()) or re.fullmatch(r'[A-Za-z]+', ref.strip()):
        return extracted, None
    # Only the balanced answer is parsed; never search unfinished reasoning for
    # an answer. Optional host dependency, with no string/number fallback parser.
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        options = dict(extraction_config=[LatexExtractionConfig()], fallback_mode='no_fallback')
        gold = parse('$$' + ref.strip() + '$$', **options)
        predicted = parse('$$' + extracted.strip() + '$$', **options)
        if len(gold) == len(predicted) == 1:
            return extracted, bool(verify(gold, predicted))
    except Exception:
        pass  # Missing parser, timeout or unsupported syntax remains ungraded.
    return extracted, None


def uniforms(identifier, sample, seed, length):
    digest = hashlib.sha256(json.dumps([identifier, sample, seed], separators=(',', ':')).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], 'big')).random(length)


def paired_accuracy_counts(original, edited):
    """Every unparsed output is noncorrect in accuracy and its paired transitions."""
    if [(r['id'], r['sample']) for r in original] != [(r['id'], r['sample']) for r in edited]:
        raise ValueError('paired identity mismatch')
    return {
        'wrong_to_right': sum(x['grade'] is not True and y['grade'] is True for x,y in zip(original,edited)),
        'right_to_wrong': sum(x['grade'] is True and y['grade'] is not True for x,y in zip(original,edited)),
        'unparsed_to_correct': sum(x['grade'] is None and y['grade'] is True for x,y in zip(original,edited)),
        'correct_to_unparsed': sum(x['grade'] is True and y['grade'] is None for x,y in zip(original,edited))}


def sample_tokens(logits, seen, random_values, sampling):
    import torch
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite generation logits')
    values = (logits.float() - seen * sampling['presence_penalty']) / sampling['temperature']
    values, indices = values.topk(min(sampling['top_k'], values.shape[-1]), dim=-1)
    probs = values.softmax(-1)
    cumulative = probs.cumsum(-1)
    probs = probs.masked_fill(cumulative - probs >= sampling['top_p'], 0)
    probs = probs / probs.sum(-1, keepdim=True)
    u = torch.as_tensor(random_values, device=logits.device, dtype=torch.float64)
    choices = (probs.double().cumsum(-1) < u[:, None]).sum(-1).clamp_max(probs.shape[-1]-1)
    return indices.gather(1, choices[:, None]).squeeze(1)


class Artifacts:
    def __init__(self, root, directory, limit):
        self.root, self.directory, self.limit = root, directory, limit
        self.manifest = []
        self.used = sum(core.local(root, str(p)).stat().st_size for p in directory.iterdir() if p.is_file())

    def reserve(self, size):
        # Keep terminal states, snapshot, result and bounded log within the job budget.
        current = sum(core.local(self.root, str(p)).stat().st_size for p in self.directory.iterdir() if p.is_file())
        if max(self.used, current) + size + 65536 > self.limit:
            raise ValueError('artifact byte budget exceeded')

    def save(self, value, kind, model, **metadata):
        array = isinstance(value, np.ndarray)
        if array:
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('artifacts must be finite float32 arrays')
            size = value.nbytes + 256
        else:
            payload = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
            size = len(payload)
        self.reserve(size)
        p = core.local(self.root, str(self.directory / (uuid.uuid4().hex + ('.npy' if array else '.json'))))
        temp = core.local(self.root, str(p) + '.tmp')
        try:
            with temp.open('xb') as f:
                if array:
                    np.save(f, value, allow_pickle=False)
                else:
                    f.write(payload)
            temp.replace(p)
        finally:
            temp.unlink(missing_ok=True)
        actual = p.stat().st_size
        self.used += actual
        entry = {'path': str(p.relative_to(self.root)), 'bytes': actual, 'sha256': core.sha256(p), 'kind': kind, 'model': model, **metadata}
        if array:
            entry.update(shape=list(value.shape), dtype=str(value.dtype))
        self.manifest.append(entry)
        return entry['path']


class Observations:
    """Hooks only read. Batch state supplies actual unpadded token coordinates."""
    def __init__(self, model, spec, artifacts, which):
        import torch
        self.spec, self.artifacts, self.which = spec, artifacts, which
        self.handles, self.rows, self.called = [], {}, {}
        self.batch = []
        self.phase, self.step = 'teacher', 0
        self.active = None
        modules = dict(model.named_modules())
        try:
            for key, suffix in [('layers', ''), ('mlp_layers', '.mlp.down_proj')]:
                for i in spec.get(key, []):
                    name = f'model.layers.{i}' + suffix
                    module = modules.get(name)
                    if module is None or (suffix and not isinstance(module, torch.nn.Linear)):
                        raise ValueError('unsupported observation module: ' + name)
                    width = module.in_features if suffix else getattr(model.config, 'hidden_size', None)
                    if type(width) is not int or width < 1:
                        raise ValueError('cannot bound observation width')
                    self.called[name] = {'class': type(module).__name__, 'width': width, 'calls': 0}
                    if suffix:
                        self.handles.append(module.register_forward_pre_hook(lambda m, args, n=name: self.capture(n, args[0])))
                    else:
                        self.handles.append(module.register_forward_hook(lambda m, args, out, n=name: self.capture(n, out[0] if isinstance(out, tuple) else out)))
        except BaseException:
            self.close()
            raise

    def estimate(self, examples, n, generate):
        count = sum((2 if generate else len(self.spec.get('positions', {}).get(e['id'], [0])) + bool(self.spec.get('mean_answer_tokens', 0))) for e in examples) * n
        self.artifacts.reserve(count * sum(m['width'] for m in self.called.values()) * 4 + count * 256)

    def capture(self, name, tensor):
        import torch
        info = self.called[name]
        if tensor.ndim != 3 or tensor.shape[0] != len(self.batch) or tensor.shape[-1] != info['width']:
            raise ValueError('unexpected observed module shape')
        info['calls'] += 1
        for i, row in enumerate(self.batch):
            key = (name, row['id'], row.get('sample', 0))
            dest = self.rows.setdefault(key, {'selected': None, 'sum': None, 'count': 0, 'positions': []})
            if self.phase == 'teacher':
                length = len(row['tokens']) - 1
                positions = self.spec.get('positions', {}).get(row['id'], [length-1])
                if any(p >= length for p in positions):
                    raise ValueError('prediction position outside sequence')
                dest['positions'] = positions
                dest['selected'] = tensor[i, positions].float().clone()
                count = min(length, self.spec.get('mean_answer_tokens', 0))
                if count:
                    dest['sum'], dest['count'] = tensor[i, :count].float().sum(0), count
            elif self.phase == 'prompt':
                dest['selected'] = tensor[i, -1:].float().clone()
                dest['positions'] = [len(row['tokens'])-1]
            elif self.active[i] and self.step < self.spec.get('mean_answer_tokens', 0):
                v = tensor[i, -1].float()
                dest['sum'] = v.clone() if dest['sum'] is None else dest['sum'] + v
                dest['count'] += 1

    def flush(self):
        for (name, identifier, sample), row in self.rows.items():
            meta = {'module': name, 'id': identifier, 'sample': sample, 'positions': row['positions'], 'mean_token_count': row['count']}
            if row['selected'] is not None:
                self.artifacts.save(row['selected'].cpu().numpy(), 'observations', self.which, **meta)
            if row['count']:
                self.artifacts.save((row['sum'] / row['count']).cpu().numpy(), 'observation_mean', self.which, **meta)
        self.rows.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def tokenize(tokenizer, examples, mode, prompt_suffix="\nReturn the final answer in \\boxed{...}."):
    result = []
    for row in examples:
        if 'tokens' in row:
            ids = row['tokens']
        elif mode == 'teacher_forced':
            if not isinstance(row.get('reference'), str) or not row['reference']:
                raise ValueError('teacher_forced requires tokens or reference text')
            ids = tokenizer.encode(row['reference'], add_special_tokens=False)
        else:
            prompt = row['prompt'] + prompt_suffix
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
        result.append({**row, 'tokens': list(ids)})
    return result


def validate_tokens(model, rows, c, spec):
    context = ev._context_limit(model)
    total = 0
    for row in rows:
        tokens = row['tokens']
        extra = spec['max_new_tokens'] if spec['type'] == 'generate' else 0
        if len(tokens) < (2 if spec['type'] == 'teacher_forced' else 1) or len(tokens) > c['max_input_tokens'] or len(tokens) + extra > context:
            raise ValueError('sequence exceeds input/context budget')
        if any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in tokens):
            raise ValueError('token outside vocabulary')
        total += len(tokens) * spec['n']
    if total > c['max_total_input_tokens']:
        raise ValueError('total input token budget exceeded')


def generate(model, tokenizer, examples, c, spec, obs):
    import torch
    device = next(model.parameters()).device
    maximum = spec['max_new_tokens']
    rows = [{**e, 'sample': s} for e in examples for s in range(spec['n'])]
    eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    if eos is None:
        eos = getattr(model.config, 'eos_token_id', None)
    eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    pad = getattr(tokenizer, 'pad_token_id', None)
    pad = pad if pad is not None else (next(iter(eos)) if eos else 0)
    output, calls, pending_bytes = [], 0, 0
    prefill_seconds, decode_seconds = 0., 0.
    keep = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    with torch.inference_mode():
        for start in range(0, len(rows), c.get('batch_size', 8)):
            chunk = rows[start:start+c.get('batch_size', 8)]
            width = max(len(e['tokens']) for e in chunk)
            if obs.artifacts is not None:
                # Bound raw token/text output before creating this batch's streams.
                obs.artifacts.reserve(pending_bytes + len(chunk) * (maximum * 32 + width * 12 + 2048))
            ids = torch.tensor([[pad]*(width-len(e['tokens'])) + e['tokens'] for e in chunk], device=device)
            mask = torch.tensor([[0]*(width-len(e['tokens'])) + [1]*len(e['tokens']) for e in chunk], device=device)
            positions = (mask.cumsum(-1)-1).clamp_min(0)
            randoms = np.stack([uniforms(e['id'], e['sample'], c['sampling']['seed'], maximum) for e in chunk])
            seen = torch.zeros((len(chunk), model.config.vocab_size), device=device, dtype=torch.bool)
            active = torch.ones(len(chunk), device=device, dtype=torch.bool)
            sequences = [[] for _ in chunk]
            reasons = ['max_new_tokens'] * len(chunk)
            obs.batch, obs.phase = chunk, 'prompt'
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timing = time.monotonic()
            out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, **keep)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            prefill_seconds += time.monotonic()-timing
            timing = time.monotonic()
            calls += 1
            past = getattr(out, 'past_key_values', None)
            if past is None:
                raise ValueError('generation backend requires actual KV cache')
            logits = out.logits[:, -1]
            del out
            for step in range(maximum):
                chosen = sample_tokens(logits, seen, randoms[:, step], c['sampling'])
                was_active = active.clone()
                chosen = torch.where(was_active, chosen, pad)
                # One GPU status transfer per step, no per-row GPU scalar reads
                # or writes. Presence applies only to actually generated tokens.
                seen.scatter_(1, chosen[:, None], seen.gather(1, chosen[:, None]) | was_active[:, None])
                is_eos = torch.zeros_like(active)
                for eos_id in eos:
                    is_eos |= chosen == eos_id
                active = was_active & ~is_eos
                status = torch.stack([chosen, was_active.long(), active.long()], -1).cpu().tolist()
                any_active = False
                active_before = []
                for i, (token, was_live, live) in enumerate(status):
                    active_before.append(bool(was_live))
                    any_active |= bool(live)
                    if was_live:
                        sequences[i].append(token)
                        if not live:
                            reasons[i] = 'eos'
                need_observe = bool(obs.called) and step < obs.spec.get('mean_answer_tokens', 0)
                if (any_active and step + 1 < maximum) or need_observe:
                    mask = torch.cat([mask, was_active.long()[:, None]], -1)
                    obs.phase, obs.step, obs.active = 'generated', step, active_before
                    out = model(input_ids=chosen[:, None], attention_mask=mask, position_ids=(mask.sum(-1)-1).clamp_min(0)[:, None], past_key_values=past, use_cache=True, **keep)
                    calls += 1
                    past, logits = out.past_key_values, out.logits[:, -1]
                    del out
                if not any_active:
                    break
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            decode_seconds += time.monotonic()-timing
            for e, tokens, reason in zip(chunk, sequences, reasons):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                extracted, correct = grade(text, e.get('answer', e.get('reference', '')))
                output.append({'id': e['id'], 'sample': e['sample'], 'input_tokens': len(e['tokens']), 'input_ids': e['tokens'], 'continuation_ids': tokens, 'text': text, 'generated_tokens': len(tokens), 'finish_reason': reason, 'extracted_answer': extracted, 'grade': correct})
            pending_bytes = len(json.dumps(output, ensure_ascii=True).encode())
            obs.flush()
    count = len(output)
    return output, {'accuracy': sum(r['grade'] is True for r in output)/count, 'parse_failure': sum(r['grade'] is None for r in output), 'cutoff': sum(r['finish_reason'] == 'max_new_tokens' for r in output), 'input_tokens': sum(r['input_tokens'] for r in output), 'generated_tokens': sum(r['generated_tokens'] for r in output), 'samples': count, 'mean_length': sum(r['generated_tokens'] for r in output)/count,
                    'prefill_seconds':prefill_seconds,'decode_seconds':decode_seconds}, calls


# Bounds include float32 logits plus reduction temporaries, independent of T.
TEACHER_HEAD_BYTES = 64 * 1024 * 1024
TEACHER_HEAD_TOKENS = 128


def teacher(model, rows, artifacts, which, obs):
    import torch
    results, calls = [], 0
    execution = {'causal_lm_forward': 0, 'backbone_forward': 0, 'output_head_forward': 0,
                 'largest_head_chunk_tokens': 0}
    total_loss, total_correct, total_tokens = 0., 0, 0
    backbone = getattr(model, 'base_model', None)
    get_head = getattr(model, 'get_output_embeddings', None)
    head = get_head() if callable(get_head) else None
    chunked = backbone is not None and backbone is not model and callable(head)
    # Qwen text causal LM applies lm_head directly to base_model hidden states.
    # Other architectures may have an extra logit transform; fail closed for
    # large unsupported models instead of silently approximating their forward.
    supported = getattr(model.config, 'model_type', '') == 'qwen3_5_text'
    chunked = chunked and supported
    vocab = model.config.vocab_size
    chunk_size = min(TEACHER_HEAD_TOKENS, TEACHER_HEAD_BYTES // (max(1, vocab) * 4 * 3))
    if chunk_size < 1:
        raise ValueError('vocabulary exceeds bounded teacher head memory')
    with torch.inference_mode():
        for row in rows:
            length = len(row['tokens']) - 1
            artifacts.reserve(length * 4 + 256)
            batch = torch.tensor([row['tokens']], device=next(model.parameters()).device)
            obs.batch, obs.phase = [row], 'teacher'
            if chunked:
                out = backbone(input_ids=batch[:, :-1], use_cache=False, return_dict=True)
                execution['backbone_forward'] += 1
                calls += 1
                hidden = out.last_hidden_state
                if hidden.ndim != 3 or hidden.shape[:2] != (1, length):
                    raise ValueError('unexpected teacher backbone shape')
                del out
            else:
                if length * vocab * 4 * 3 > TEACHER_HEAD_BYTES:
                    raise ValueError('unsupported model requires bounded backbone/head teacher path')
                out = model(input_ids=batch[:, :-1], use_cache=False)
                execution['causal_lm_forward'] += 1
                calls += 1
            # Only O(T) target scores leave GPU; never retain T*vocab logits.
            scores = np.empty(length, dtype=np.float32)
            correct = 0
            for start in range(0, length, chunk_size):
                end = min(length, start + chunk_size)
                if chunked:
                    raw = head(hidden[:, start:end])[0]
                    calls += 1
                    execution['output_head_forward'] += 1
                    execution['largest_head_chunk_tokens'] = max(execution['largest_head_chunk_tokens'], end-start)
                else:
                    raw = out.logits[0, start:end]
                logits = raw.float()
                if logits.shape != (end-start, vocab) or not torch.isfinite(logits).all():
                    raise ValueError('invalid teacher logits')
                targets = batch[0, start+1:end+1]
                lp = logits.gather(1, targets[:, None]).squeeze(1) - logits.logsumexp(-1)
                scores[start:end] = lp.cpu().numpy()
                correct += int((logits.argmax(-1) == targets).sum())
                del raw, logits, lp
            if chunked:
                del hidden
            else:
                del out
            nll = -float(scores.astype(np.float64).sum())
            path = artifacts.save(scores, 'target_logprobs', which, id=row['id'])
            results.append({'id': row['id'], 'tokens': length, 'nll': nll/length, 'accuracy': correct/length, 'logprobs_path': path})
            total_loss += nll
            total_correct += correct
            total_tokens += length
            obs.flush()
    artifacts.save(results, 'teacher_metrics', which)
    return {'nll': total_loss/total_tokens, 'accuracy': total_correct/total_tokens, 'target_tokens': total_tokens, 'input_tokens': sum(len(r['tokens']) for r in rows), 'generated_tokens': 0, 'samples': len(rows), 'execution': execution}, calls


def cast_for_inference(model, device, dtype, source_dtypes=None):
    """Cast inference matrices while preserving source FP32 parameters and positional buffers."""
    import torch
    named = dict(model.named_parameters())
    retained = {name: named[name].detach().clone().float() for name, original in (source_dtypes or {}).items()
                if original == 'torch.float32' and name in named}
    missing = [name for name, original in (source_dtypes or {}).items() if original == 'torch.float32' and name not in named]
    if missing:
        raise ValueError('source FP32 parameter mapping missing in loaded model')
    buffers, identities = {}, {}
    for name, value in model.named_buffers(remove_duplicate=False):
        if value.dtype == torch.float32:
            identities.setdefault(id(value), value.detach().clone())
            buffers[name] = id(value)
    model.to(device=device, dtype=dtype)
    modules = dict(model.named_modules())
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in retained.items():
            named[name].data = value.to(device=device)
        identities = {key: value.to(device=device) for key,value in identities.items()}
        for name, key in buffers.items():
            owner, _, leaf = name.rpartition('.')
            modules[owner]._buffers[leaf] = identities[key]
    model.eval().requires_grad_(False)
    model._research_precision_policy = {'matrix_dtype': str(dtype), 'source_fp32_parameters': sorted(retained),
        'preserved_fp32_buffers': sorted(buffers), 'reason': 'Preserve published FP32 values and rotary precision; no parameter fitting.'}
    return model


def load_verified(root, c, spec, which):
    import torch
    candidate = spec.get('candidate_dir') if which == 'edited' else None
    model = inference.load_inference_model(root, c['model_dir'], candidate, device='cpu')
    dtype = getattr(torch, c['inference_dtype'])
    source_dtypes_path = core.local(root, str(Path(c['model_dir']) / 'source_dtypes.json'))
    source_dtypes = api.read(root, source_dtypes_path) if source_dtypes_path.exists() else {}
    model = cast_for_inference(model, c['device'], dtype, source_dtypes)
    actual = 0
    if candidate:
        manifest, patch = ev.verify_candidate(root, c['model_dir'], candidate)
        state = model.state_dict()
        with safe_open(core.local(root, str(Path(c['model_dir']) / 'model.safetensors')), framework='np') as source:
            for name, array in patch.items():
                expected = torch.from_numpy(array).to(device=c['device'], dtype=state[name].dtype)
                original = torch.from_numpy(source.get_tensor(name)).to(device=c['device'], dtype=state[name].dtype)
                if name not in state or not torch.equal(state[name], expected):
                    raise ValueError('actual cast overlay differs from verified candidate')
                actual += int((expected != original).sum())
        if manifest['report']['changed_parameters'] and actual == 0:
            raise ValueError('nonidentity edit disappears under inference dtype cast')
    return model, actual


def vllm_generate(root, jid, tokenizer, rows, c, spec, artifacts, which):
    """Optimized engine subprocess; all model execution stays inside this MCP job."""
    inputs = artifacts.save(rows, 'generation_inputs', which)
    raw_path = str(api.jobdir(root,jid).relative_to(root)/('vllm-'+which+'.json'))
    command=[c['vllm_python'],'-B','-m','weight_space_lab.vllm_generation',
        '--workspace',str(root),'--job',jid,'--which',which,'--inputs',inputs,'--output',raw_path]
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]),
         'VLLM_WORKER_MULTIPROC_METHOD':'spawn','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PATH':str(Path(c['vllm_python']).parent)+os.pathsep+os.environ.get('PATH','')}
    started=time.monotonic()
    subprocess.run(command,check=True,env=env)
    payload=api.read(root,raw_path)
    verification=payload['verification']
    if verification['requires_grad']:
        raise ValueError('engine parameters are not frozen')
    ordered=[(r,s) for r in rows for s in range(spec['n'])]
    if len(payload['rows']) != len(ordered):
        raise ValueError('engine sample count mismatch')
    output=[]
    for (example,sample),r in zip(ordered,payload['rows']):
        if (r['id'],r['sample']) != (example['id'],sample):
            raise ValueError('engine sample identity mismatch')
        tokens=r['continuation_ids']
        if not isinstance(tokens,list) or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError('invalid actual engine continuation')
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        extracted,correct=grade(text,example.get('answer',''))
        output.append({'id':example['id'],'sample':sample,'input_tokens':len(example['tokens']),
            'input_ids':example['tokens'],'continuation_ids':tokens,'text':text,'generated_tokens':len(tokens),
            'finish_reason':r['finish_reason'],'extracted_answer':extracted,'grade':correct})
    artifacts.save(payload,'engine_evidence',which)
    core.local(root,raw_path).unlink()  # the same raw values now have a manifest-bound copy
    n=len(output)
    metrics={'accuracy':sum(r['grade'] is True for r in output)/n,'parse_failure':sum(r['grade'] is None for r in output),
        'cutoff':sum(r['finish_reason']=='max_new_tokens' for r in output),'samples':n,
        'input_tokens':sum(r['input_tokens'] for r in output),'generated_tokens':sum(r['generated_tokens'] for r in output),
        'mean_length':sum(r['generated_tokens'] for r in output)/n,'engine_seconds':time.monotonic()-started,
        'backend':payload['backend'],'model_forward_counter_scope':payload['model_forward_counter_scope']}
    return output,metrics,verification['actual_changed']


def execute(root, jid):
    import torch
    started = time.monotonic()
    d = api.jobdir(root, jid)
    snap = api.load_snapshot(root, jid)
    c, spec = snap['config'], snap['spec']
    api.verify_hashes(root, snap['hashes'])
    spec, examples = api.validate_spec(root, c, spec)
    # All tokenizer files were checked and bound before loading; no remote code/network.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(core.local(root, c['model_dir'])), local_files_only=True, trust_remote_code=False)
    rows = tokenize(tokenizer, examples, spec['type'], c.get('prompt_suffix', '\nReturn the final answer in \\boxed{...}.'))
    artifacts = Artifacts(root, d, c['max_artifact_bytes_per_job'])
    metrics, outputs, changes, observations, precision_policies = {}, {}, {}, {}, {}
    calls = 0
    for which in (['original', 'edited'] if spec['model'] == 'both' else [spec['model']]):
        api.verify_hashes(root, snap['hashes'])
        if spec['type']=='generate' and c.get('generation_backend')=='vllm':
            generated,metrics[which],changes[which]=vllm_generate(root,jid,tokenizer,rows,c,spec,artifacts,which)
            outputs[which]=generated
            artifacts.save(generated,'generations',which)
            observations[which]={}
            precision_policies[which]={'backend':'vllm','checkpoint_precision':'FP32 source; engine BF16 compute with native model precision policy; actual target overlays verified on GPU.'}
            # The engine owns internal graph invocations; do not call a submission
            # one network forward. The terminal accounting distinguishes them.
            api.verify_hashes(root,snap['hashes'])
            continue
        model, changes[which] = load_verified(root, c, spec, which)
        precision_policies[which] = getattr(model, '_research_precision_policy', {})
        validate_tokens(model, rows, c, spec)
        obs = Observations(model, spec.get('observations', {}), artifacts, which)
        try:
            obs.estimate(rows, spec['n'], spec['type'] == 'generate')
            if spec['type'] == 'generate':
                outputs[which], metrics[which], count = generate(model, tokenizer, rows, c, spec, obs)
                artifacts.save(outputs[which], 'generations', which)
            else:
                metrics[which], count = teacher(model, rows, artifacts, which, obs)
            calls += count
            observations[which] = obs.called
        finally:
            obs.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        api.verify_hashes(root, snap['hashes'])
    if spec['model'] == 'both':
        a, b = outputs['original'], outputs['edited']
        if [(r['id'], r['sample']) for r in a] != [(r['id'], r['sample']) for r in b]:
            raise ValueError('paired identity mismatch')
        metrics['paired'] = paired_accuracy_counts(a, b)
    result = {'job_id': jid, 'provenance': snap['hashes'], 'metrics': metrics, 'actual_changed': changes, 'observed_modules': observations, 'precision_policies':precision_policies,'model_forward_invocations': calls, 'successful_samples': sum(m['samples'] for m in metrics.values() if 'samples' in m), 'wall_seconds': time.monotonic()-started, 'logical_passes': 1, 'artifacts': artifacts.manifest, 'gradients_enabled': False, 'optimizer_steps': 0}
    if spec['type']=='generate' and c.get('generation_backend')=='vllm':
        result.update(model_forward_invocations=None,engine_generation_calls=len(outputs),execution_counter_scope='engine_submissions_only')
    if len(json.dumps(result).encode()) > 14000:
        # Full hashes/shapes remain in a hash-bound external manifest. Keep the
        # terminal API compact even when selecting many per-item observations.
        full = list(artifacts.manifest)
        artifacts.save(full, 'artifact_manifest', 'all')
        descriptor = artifacts.manifest[-1]
        result['artifact_manifest'] = descriptor
        result['artifact_count'] = len(full)
        result['artifacts'] = [a for a in full if a['kind'] == 'generations'] + [descriptor]
        result['array_layouts'] = sorted({(a['kind'], a.get('dtype', ''), tuple(a.get('shape', []))) for a in full if 'shape' in a})
    if len(json.dumps(result).encode()) > 16384:
        raise ValueError('compact summary budget exceeded')
    api.verify_hashes(root, snap['hashes'])
    api.atomic(root, d / 'result.json', result)
    api.verified_result(root, jid)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--execute', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.workspace.resolve(strict=True)
    d = api.jobdir(root, args.job)
    if not args.execute:
        # This process (the ledger PID) owns capture until the execution child
        # exits, even when the submitting MCP server has already disappeared.
        log = core.local(root, str(d / 'worker.log'))
        child = subprocess.Popen([sys.executable, '-B', '-m', 'weight_space_lab.research_worker',
                                  '--workspace', str(root), '--job', args.job, '--execute'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        capture = api.drain_worker(child, log)
        api.atomic(root, d / 'log_capture.json', capture)
        return child.returncode
    claimed = False
    try:
        # Synchronize with parent's durable PID registration and reject replay.
        with api.ledger(root) as db:
            row = db.execute('SELECT status FROM jobs WHERE id=?', (args.job,)).fetchone()
            if row is None or row[0] != 'pending':
                raise ValueError('job not pending')
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (args.job,))
            claimed = True
            api.atomic(root, d / 'state.json', {'status': 'running'})
        tripwires()
        # Limit library logging, which shares the job's bounded artifact allocation.
        import logging
        logging.disable(logging.CRITICAL)
        import resource
        c = api.config(root)
        # Keep the job-output soft limit. The controlled engine subprocess may
        # temporarily raise it only while materializing a separately budgeted
        # immutable checkpoint, then restores it before inference/output.
        _,hard=resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(resource.RLIMIT_FSIZE, (c['max_artifact_bytes_per_job'], hard))
        execute(root, args.job)
        api.atomic(root, d / 'state.json', {'status': 'succeeded'})
    except BaseException as error:
        if not claimed:
            return 1
        api.atomic(root, d / 'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {str(error)[:1500]}'})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
