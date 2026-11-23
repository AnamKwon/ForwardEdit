"""Known-value frozen fixtures exercise actual worker code, never train/download."""
import json
import io
import os
from types import SimpleNamespace
import numpy as np
import pytest
from safetensors.numpy import save_file
from weight_space_lab import core, research as r, research_worker as w


def world(root, monkeypatch, **updates):
    import torch
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = torch.nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                self.mlp.down_proj.weight.copy_(torch.eye(3))
        def forward(self, x):
            return x + self.mlp.down_proj(x)
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block()])
            self.config = SimpleNamespace(vocab_size=5, n_positions=32, hidden_size=3, eos_token_id=4)
            self.seen_calls = []
            self.eval().requires_grad_(False)
        def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
            assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
            assert not self.training and all(not p.requires_grad for p in self.parameters())
            self.seen_calls.append((input_ids.clone(), attention_mask, position_ids, past_key_values))
            x = torch.nn.functional.one_hot(input_ids % 3, 3).float()
            x = self.model.layers[0](x)
            # Actual frozen module values affect logits; stochastic outcomes are possible.
            logits = torch.cat([x, x[..., :2] * .5], -1)
            return SimpleNamespace(logits=logits, past_key_values=(0 if past_key_values is None else past_key_values)+input_ids.shape[1] if use_cache else None)
    class Tokenizer:
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 1]
        def apply_chat_template(self, messages, **kw):
            assert kw['enable_thinking'] and '\\boxed' in messages[0]['content']
            return [1, 2]
        def decode(self, tokens, **kw):
            return '\\boxed{' + str(sum(tokens)) + '}'
    (root / 'model').mkdir()
    (root / 'model/config.json').write_text('{}')
    model = LM()
    save_file({k: v.numpy() for k, v in model.state_dict().items()}, root / 'model/model.safetensors')
    (root / 'data.json').write_text(json.dumps({'splits': {'discovery': [
        {'id': 'a', 'prompt': 'a', 'answer': '4', 'reference': 'solution'},
        {'id': 'b', 'tokens': [2, 1, 2], 'answer': '5'}], 'validation': [{'id': 'v', 'tokens': [1, 2]}]}}))
    c = dict(model_dir='model', source_sha256=core.sha256(root/'model/model.safetensors'), config_sha256=core.sha256(root/'model/config.json'), dataset_path='data.json', dataset_sha256=core.sha256(root/'data.json'), successful_forward_budget=50, attempt_budget=75, device='cpu', inference_dtype='float32', max_items=128, max_new_tokens=16, max_input_tokens=32, max_total_input_tokens=128, sampling=dict(temperature=.6, top_p=.95, top_k=5, presence_penalty=1.5, seed=4217), max_artifact_bytes_per_job=2**20, max_artifact_bytes_total=10*2**20, batch_size=2)
    c.update(updates)
    (root/'research.json').write_text(json.dumps(c))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    models = []
    def load(root, model_dir, candidate_dir=None, **kw):
        m = LM()
        if candidate_dir:
            _, patch = r.ev.verify_candidate(root, model_dir, candidate_dir)
            with torch.no_grad():
                for name, a in patch.items():
                    m.state_dict()[name].copy_(torch.from_numpy(a))
        models.append(m)
        return m
    monkeypatch.setattr(w.inference, 'load_inference_model', load)
    return c, models, Tokenizer()


def spec(**kwargs):
    return dict(type='generate', model='original', split='discovery', ids=['a', 'b'], n=1, max_new_tokens=4, **kwargs)


def test_image_bytes_are_bound_to_forward_snapshot(tmp_path, monkeypatch):
    c, _, _ = world(tmp_path, monkeypatch)
    image = tmp_path / 'image.png'
    image.write_bytes(b'original image fixture bytes')
    data = json.loads((tmp_path / 'data.json').read_text())
    data['splits']['discovery'][0]['images'] = ['image.png']
    (tmp_path / 'data.json').write_text(json.dumps(data))
    c['dataset_sha256'] = core.sha256(tmp_path / 'data.json')
    c['multimodal'] = {'kind': 'gemma3', 'vision_checkpoint': 'model/model.safetensors',
                       'config': 'model/config.json', 'processor_dir': 'model'}
    (tmp_path / 'research.json').write_text(json.dumps(c))
    r.validate_spec(tmp_path, c, spec())
    hashes = r.snapshot(tmp_path, c, spec())
    assert hashes['image.png'] == core.sha256(image)
    image.write_bytes(b'changed image fixture bytes')
    with pytest.raises(ValueError, match='provenance hash mismatch: image.png'):
        r.verify_hashes(tmp_path, hashes)


def reserve(root, monkeypatch, request):
    class Process:
        pid = os.getpid()
        def __init__(self, argv, **kw):
            self.stdout = io.BytesIO()
            assert argv[1:4] == ['-B', '-m', 'weight_space_lab.research_worker']
        def wait(self):
            return 0
    monkeypatch.setattr(r, 'subprocess', SimpleNamespace(Popen=Process, DEVNULL=-3, PIPE=-1, STDOUT=-2))
    return r.research_forward(root, request)['job_id']


def finish(root, jid):
    result = w.execute(root, jid)
    r.atomic(root, r.jobdir(root, jid)/'state.json', {'status': 'succeeded'})
    return result


def test_actual_generation_observations_and_provenance(tmp_path, monkeypatch):
    c, models, _ = world(tmp_path, monkeypatch)
    jid = reserve(tmp_path, monkeypatch, spec(observations={
        'layers': [0], 'mlp_layers': [0], 'mean_answer_tokens': 3, 'first_token_ids': [2, 0],
    }))
    result = finish(tmp_path, jid)
    assert result['model_forward_invocations'] >= 2
    assert result['successful_samples'] == 2 and result['optimizer_steps'] == 0
    assert result['observed_modules']['original']['model.layers.0']['calls'] >= 2
    assert all(not m.model.layers[0]._forward_hooks and not m.model.layers[0].mlp.down_proj._forward_pre_hooks for m in models)
    state = r.research_status(tmp_path, jid)
    assert state['successful_completed'] == 1
    assert state['progress'] == {
        'model': 'original', 'completed_samples': 2, 'total_samples': 2,
        'generated_tokens': result['metrics']['original']['generated_tokens'],
    }
    assert r.research_status(tmp_path, jid)['successful_completed'] == 1
    rows = r.research_read_generations(tmp_path, jid, ['a'], max_chars=4)['rows']
    assert len(rows) == 1 and len(rows[0]['text']) <= 4
    arrays = [a for a in result['artifacts'] if a['kind'] == 'observations']
    a = r.research_read_array(tmp_path, arrays[0]['path'], [[0, 1]])
    assert a['entries'][0]['value'] in (0., 2.)
    first = next(a for a in result['artifacts'] if a['kind'] == 'first_token_logits')
    assert first['token_ids'] == [2, 0] and first['model'] == 'original'
    assert json.loads((tmp_path / first['path']).read_text()) == [
        {'id': 'a', 'sample': 0, 'logits': [2.0, 0.0]},
        {'id': 'b', 'sample': 0, 'logits': [2.0, 0.0]},
    ]
    (tmp_path/'model/config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        r.research_result(tmp_path, jid)


def test_teacher_actual_nll_and_selected_positions(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    request = {'type': 'teacher_forced', 'model': 'original', 'split': 'discovery', 'ids': ['b'], 'observations': {'layers': [0], 'positions': {'b': [0, 1]}, 'mean_answer_tokens': 2}}
    jid = reserve(tmp_path, monkeypatch, request)
    result = finish(tmp_path, jid)
    a = next(a for a in result['artifacts'] if a['kind'] == 'target_logprobs')
    logp = np.load(tmp_path/a['path'], allow_pickle=False)
    expected = np.array([0., 0., 2., 0., 0.])
    expected = expected - np.log(np.exp(expected).sum())
    assert logp.shape == (2,) and logp.dtype == np.float32
    np.testing.assert_allclose(logp[0], expected[1], rtol=1e-6)
    assert result['metrics']['original']['target_tokens'] == 2
    assert result['model_forward_invocations'] == 1


def test_serial_failure_retry_and_budget(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch, successful_forward_budget=1)
    jid = reserve(tmp_path, monkeypatch, spec())
    with pytest.raises(ValueError, match='busy'):
        r.research_forward(tmp_path, spec())
    r.atomic(tmp_path, r.jobdir(tmp_path, jid)/'state.json', {'status': 'failed', 'error': 'OOM'})
    assert r.research_status(tmp_path, jid)['successful_completed'] == 0
    other = reserve(tmp_path, monkeypatch, spec())
    finish(tmp_path, other)
    assert r.research_status(tmp_path, other)['attempt_number'] == 2
    with pytest.raises(ValueError, match='exhausted'):
        r.research_forward(tmp_path, spec())


def test_death_and_forged_success_fail(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    jid = reserve(tmp_path, monkeypatch, spec())
    with r.ledger(tmp_path) as db:
        db.execute('UPDATE jobs SET pid=99999999,birth=? WHERE id=?', ('absent', jid))
    state = r.research_status(tmp_path, jid)
    assert state['status'] == 'failed' and state['successful_completed'] == 0
    jid = reserve(tmp_path, monkeypatch, spec())
    r.atomic(tmp_path, r.jobdir(tmp_path, jid)/'state.json', {'status': 'succeeded'})
    assert r.research_status(tmp_path, jid)['status'] == 'failed'


@pytest.mark.parametrize('change', [
    {'split': 'test'}, {'ids': ['a', 'a']}, {'ids': ['unknown']}, {'max_new_tokens': 100},
    {'dataset_path': '../escape'}, {'model': 'edited'}, {'n': 5}, {'observations': {'mean_answer_tokens': 8193}},
    {'observations': {'first_token_ids': [2, 2]}},
    {'type': 'teacher_forced', 'observations': {'first_token_ids': [2]}},
    {'observations': {'generation_logits': {'token_ids': [2], 'positions': {'a': [4]}}}},
    {'first_token_logit_bias': []},
    {'first_token_logit_bias': [{'token_id': 1, 'bias': 101}]},
    {'first_token_logit_bias': [{'token_id': True, 'bias': 1}]},
    {'first_token_logit_bias': [{'token_id': 1, 'bias': True}]},
    {'first_token_logit_bias': [{'token_id': 1, 'bias': 1}, {'token_id': 1, 'bias': -1}]},
    {'type': 'teacher_forced', 'first_token_logit_bias': [{'token_id': 1, 'bias': 1}]},
])
def test_reject_invalid_specs(tmp_path, monkeypatch, change):
    world(tmp_path, monkeypatch)
    request = spec()
    request.update(change)
    with pytest.raises((ValueError, FileNotFoundError)):
        r.research_forward(tmp_path, request)


def test_override_provenance_and_context(tmp_path, monkeypatch):
    c, models, _ = world(tmp_path, monkeypatch)
    override = {'splits': {'discovery': [{'id': 'a', 'tokens': [1]*33}]}}
    (tmp_path/'contexts.json').write_text(json.dumps(override))
    jid = reserve(tmp_path, monkeypatch, {'type': 'teacher_forced', 'model': 'original', 'split': 'discovery', 'ids': ['a'], 'dataset_path': 'contexts.json'})
    with pytest.raises(ValueError, match='context budget'):
        w.execute(tmp_path, jid)
    assert not models[0].seen_calls
    override['splits']['discovery'][0]['id'] = 'invented'
    (tmp_path/'contexts.json').write_text(json.dumps(override))
    with pytest.raises(ValueError, match='provenance'):
        r.validate_spec(tmp_path, c, {'type': 'teacher_forced', 'model': 'original', 'split': 'discovery', 'ids': ['invented'], 'dataset_path': 'contexts.json'})


def test_array_bounds_and_scalar(tmp_path):
    np.save(tmp_path/'a.npy', np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    np.save(tmp_path/'s.npy', np.array(7., dtype=np.float32))
    assert r.research_read_array(tmp_path, 'a.npy', [[1, 2, 3]])['entries'][0]['value'] == 23
    assert r.research_read_array(tmp_path, 's.npy', [[]])['entries'][0]['value'] == 7
    for path, indices in [('a.npy', [[2, 0, 0]]), ('a.npy', [[0]]), ('a.npy', [[0, 0, 0]]*4097), ('../escape', [[]])]:
        with pytest.raises((ValueError, FileNotFoundError)):
            r.research_read_array(tmp_path, path, indices)
    np.save(tmp_path/'bad.npy', np.array([np.nan]))
    with pytest.raises(ValueError, match='nonfinite'):
        r.research_read_array(tmp_path, 'bad.npy', [[0]])
    np.save(tmp_path/'pickle.npy', np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        r.research_read_array(tmp_path, 'pickle.npy', [[0]])


def test_sampling_presence_and_independent_streams():
    import torch
    sampling = dict(temperature=1., top_p=1., top_k=3, presence_penalty=10.)
    logits = torch.tensor([[2., 1., 0.]])
    empty = torch.zeros_like(logits, dtype=torch.bool)
    assert w.sample_tokens(logits, empty, [0.], sampling).item() == 0
    assert w.sample_tokens(logits, empty, [.99], sampling).item() == 2
    seen = empty.clone(); seen[0, 0] = True
    assert w.sample_tokens(logits, seen, [0.], sampling).item() == 1
    np.testing.assert_array_equal(w.uniforms('a', 0, 4217, 8), w.uniforms('a', 0, 4217, 32)[:8])
    assert not np.array_equal(w.uniforms('a', 0, 4217, 8), w.uniforms('a', 1, 4217, 8))


def test_order_batch_identity_and_left_padding(tmp_path, monkeypatch):
    c, models, tokenizer = world(tmp_path, monkeypatch)
    rows = [{'id': 'a', 'tokens': [1]}, {'id': 'b', 'tokens': [2, 1]}]
    out = []
    for batch, items in [(2, rows), (1, rows[::-1])]:
        c['batch_size'] = batch
        m = w.inference.load_inference_model(tmp_path, 'model')
        obs = w.Observations(m, {}, None, 'original')
        result, _, _ = w.generate(m, tokenizer, items, c, spec(), obs)
        out.append({r['id']: r['continuation_ids'] for r in result})
    assert out[0] == out[1]
    first = models[0].seen_calls[0]
    assert first[1].tolist() == [[0, 1], [1, 1]]
    assert first[2].tolist() == [[0, 0], [0, 1]]
    assert models[0].seen_calls[1][0].shape[1] == 1


@pytest.mark.parametrize('text,answer,expected', [(r'work \boxed{\frac{1}{2}}', '.5', True), (r'\boxed{1} then \boxed{2}', '#### 2', True), (r'\boxed{3} then \boxed{4', '3', None), ('unfinished thinking 42', '42', None), (r'\boxed{x}', 'x', None), (r'\boxed{2}', '3', False)])
def test_grading(text, answer, expected):
    assert w.grade(text, answer)[1] is expected


def test_hooks_removed_and_artifact_budget(tmp_path, monkeypatch):
    c, models, _ = world(tmp_path, monkeypatch)
    jid = reserve(tmp_path, monkeypatch, spec(observations={'layers': [0, 99]}))
    with pytest.raises(ValueError, match='unsupported observation'):
        w.execute(tmp_path, jid)
    assert not models[0].model.layers[0]._forward_hooks
    a = w.Artifacts(tmp_path, r.jobdir(tmp_path, jid), 70000)
    with pytest.raises(ValueError, match='byte budget'):
        a.save(np.zeros(2000, dtype=np.float32), 'array', 'original')


def test_candidate_identity_common_seed_and_cast_loss(tmp_path, monkeypatch):
    c, _, _ = world(tmp_path, monkeypatch)
    name = 'model.layers.0.mlp.down_proj.weight'
    plan = {'checkpoint': 'model/model.safetensors', 'source_sha256': c['source_sha256'], 'hypothesis': 'explicit identity control', 'limits': {'max_changed_fraction': 1., 'max_relative_l2': 1.}, 'operations': [{'kind': 'scale', 'tensor': name, 'axis': 0, 'indices': [0], 'strength': 0.}]}
    core.write_candidate(tmp_path, plan, 'identity')
    request = spec(); request.update(model='both', candidate_dir='identity')
    jid = reserve(tmp_path, monkeypatch, request)
    result = finish(tmp_path, jid)
    gen = [r.read(tmp_path, a['path']) for a in result['artifacts'] if a['kind'] == 'generations']
    assert gen[0] == gen[1]
    assert result['actual_changed']['edited'] == 0
    plan['operations'][0]['strength'] = .00001
    core.write_candidate(tmp_path, plan, 'vanishing')
    c['inference_dtype'] = 'bfloat16'
    with pytest.raises(ValueError, match='disappears'):
        w.load_verified(tmp_path, c, {'candidate_dir': 'vanishing'}, 'edited')


def test_immutable_snapshot_and_attempt_artifact_limits(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch, attempt_budget=1)
    jid = reserve(tmp_path, monkeypatch, spec())
    path = r.jobdir(tmp_path, jid)/'snapshot.json'
    data = r.read(tmp_path, path)
    data['spec']['ids'] = ['a']
    r.atomic(tmp_path, path, data)
    with pytest.raises(ValueError, match='snapshot hash'):
        w.execute(tmp_path, jid)
    r.atomic(tmp_path, r.jobdir(tmp_path, jid)/'state.json', {'status': 'failed', 'error': 'tampered'})
    assert r.research_status(tmp_path, jid)['successful_completed'] == 0
    with pytest.raises(ValueError, match='exhausted'):
        r.research_forward(tmp_path, spec())
    c = r.read(tmp_path, 'research.json'); c['attempt_budget'] = 2
    r.atomic(tmp_path, 'research.json', c)
    with pytest.raises(ValueError, match='immutable'):
        r.research_status(tmp_path, jid)


def test_parallel_reservation_and_total_bytes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    world(tmp_path, monkeypatch)
    # Install fixture spawn stub before starting independent server threads.
    first = reserve(tmp_path, monkeypatch, spec())
    r.atomic(tmp_path, r.jobdir(tmp_path, first)/'state.json', {'status': 'failed', 'error': 'control'})
    r.research_status(tmp_path, first)
    def submit():
        try:
            return r.research_forward(tmp_path, spec())['job_id']
        except ValueError as e:
            return str(e)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit(), range(2)))
    assert sum('busy' in x for x in results) == 1
    jid = next(x for x in results if 'busy' not in x)
    r.atomic(tmp_path, r.jobdir(tmp_path, jid)/'state.json', {'status': 'failed', 'error': 'control'})
    r.research_status(tmp_path, jid)
    with r.ledger(tmp_path) as db:
        db.execute('UPDATE jobs SET bytes=? WHERE id=?', (10*2**20, jid))
    with pytest.raises(ValueError, match='total artifact'):
        r.research_forward(tmp_path, spec())


def test_worker_tripwires_and_bounded_log(tmp_path, monkeypatch):
    import torch
    # conftest restores the original attributes after this test too.
    w.tripwires()
    with pytest.raises(RuntimeError, match='forbidden'):
        torch.autograd.grad(None, None)
    with pytest.raises(RuntimeError, match='forbidden'):
        torch.optim.Optimizer([],{})
    waited = []
    process = SimpleNamespace(stdout=io.BytesIO(b'startup\n'+b'x'*100000+b'\nfinal error'), wait=lambda: waited.append(True))
    log = tmp_path/'worker.log'
    r.drain_worker(process, log)
    assert log.stat().st_size == 16384 and waited and process.stdout.closed
    assert log.read_bytes().startswith(b'startup\n') and log.read_bytes().endswith(b'\nfinal error')


def test_reference_only_override_and_hook_failure_cleanup(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    (tmp_path/'references.json').write_text(json.dumps({'splits': {'discovery': [{'id': 'a', 'reference': 'exact teacher text'}]}}))
    request = {'type': 'teacher_forced', 'model': 'original', 'split': 'discovery', 'ids': ['a'], 'dataset_path': 'references.json', 'observations': {'layers': [0], 'positions': {'a': [99]}}}
    jid = reserve(tmp_path, monkeypatch, request)
    with pytest.raises(ValueError, match='prediction position'):
        w.execute(tmp_path, jid)
    # Reconfigure the explicit observation, as a fresh failed attempt, not fitting.
    r.atomic(tmp_path, r.jobdir(tmp_path, jid)/'state.json', {'status': 'failed', 'error': 'invalid positions'})
    request['observations'] = {}
    jid = reserve(tmp_path, monkeypatch, request)
    assert finish(tmp_path, jid)['metrics']['original']['target_tokens'] == 2


def test_large_manifest_remains_hash_verified(tmp_path, monkeypatch):
    world(tmp_path, monkeypatch)
    jid = reserve(tmp_path, monkeypatch, spec())
    original = w.Artifacts.save
    def padded(self, value, kind, model, **metadata):
        # Exercise external-manifest formatting without a large array/model.
        if kind == 'generations':
            for i in range(50):
                original(self, np.ones(2, dtype=np.float32), 'diagnostic', model, description='bounded metadata'*10, ordinal=i)
        return original(self, value, kind, model, **metadata)
    monkeypatch.setattr(w.Artifacts, 'save', padded)
    result = finish(tmp_path, jid)
    assert len(json.dumps(result).encode()) <= 16384
    manifest = result['artifact_manifest']
    full = r.read(tmp_path, manifest['path'])
    assert len(full) == 51
    path = tmp_path/full[0]['path']
    path.write_bytes(b'tampered')
    assert r.research_status(tmp_path, jid)['status'] == 'failed'


def test_real_subprocess_failure_retains_attempt(tmp_path, monkeypatch):
    import time
    world(tmp_path, monkeypatch)
    # This workspace has no valid HF model/tokenizer config, so the actual
    # subprocess must fail locally before it can load any model.
    jid = r.research_forward(tmp_path, spec())['job_id']
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = r.research_status(tmp_path, jid)
        if state['status'] == 'failed':
            break
        time.sleep(.05)
    assert state['status'] == 'failed' and state['successful_completed'] == 0
    assert state['attempt_number'] == 1 and state['error']
    assert r.research_result(tmp_path, jid)['status'] == 'failed'
    assert (r.jobdir(tmp_path, jid)/'worker.log').stat().st_size <= 16384


@pytest.mark.parametrize('keep_submitter', [True, False])
def test_actual_api_detached_capture_survives_submitter_exit(tmp_path, monkeypatch, keep_submitter):
    """Actual reservation + detached supervisor; replace only inference child.

    The inert child is a lifecycle control, explicitly a failed logical forward.
    Its delayed stdout would raise BrokenPipeError with the old parent drainer.
    """
    import subprocess
    import sys
    import time
    from pathlib import Path
    world(tmp_path, monkeypatch)
    child = tmp_path/'inert.py'
    child.write_text('''import sys,time
from pathlib import Path
from weight_space_lab import research as r
root=Path(sys.argv[1]); jid=sys.argv[2]
time.sleep(.4)
print('delayed diagnostic'*10000, flush=True)
(root/'completed').write_text('stdout survived')
r.atomic(root,r.jobdir(root,jid)/'state.json',{'status':'failed','error':'inert lifecycle control'})
''')
    wrapper = tmp_path/'supervisor.py'
    wrapper.write_text('''import sys
from weight_space_lab import research_worker as w
real=w.subprocess.Popen
# Remove fixture filename so w.main sees normal --workspace/--job arguments.
fixture=sys.argv.pop(1)
def launch(argv, **kw):
    return real([sys.executable,'-B',fixture,argv[argv.index('--workspace')+1],argv[argv.index('--job')+1]], **kw)
w.subprocess.Popen=launch
raise SystemExit(w.main())
''')
    submit = tmp_path/'submit.py'
    submit.write_text('''import sys,json,subprocess
from pathlib import Path
from types import SimpleNamespace
from weight_space_lab import research as r
real=subprocess.Popen
wrapper,fixture,root,keep=sys.argv[1:]
def launch(argv, **kw):
    assert kw['start_new_session'] and kw['stdout']==subprocess.DEVNULL
    return real([sys.executable,'-B',wrapper,fixture,*argv[4:]], **kw)
r.subprocess=SimpleNamespace(Popen=launch,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT)
result=r.research_forward(Path(root),dict(type='generate',model='original',split='discovery',ids=['a'],n=1,max_new_tokens=2))
print(json.dumps(result),flush=True)
if keep=='yes': sys.stdin.readline()
''')
    env = {**os.environ, 'PYTHONPATH': str(Path.cwd())}
    parent = subprocess.Popen([sys.executable, '-B', str(submit), str(wrapper), str(child), str(tmp_path), 'yes' if keep_submitter else 'no'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = parent.stdout.readline()
        assert line, parent.stderr.read()
        jid = json.loads(line)['job_id']
        if not keep_submitter:
            assert parent.wait(timeout=5) == 0
        deadline = time.monotonic()+15
        while time.monotonic() < deadline and not (tmp_path/'completed').exists():
            time.sleep(.05)
        assert (tmp_path/'completed').read_text() == 'stdout survived'
        # Wait until the supervisor has drained and closed the log.
        while time.monotonic() < deadline:
            with r.ledger(tmp_path) as db:
                pid = db.execute('SELECT pid FROM jobs WHERE id=?', (jid,)).fetchone()[0]
            if r.birth(pid) is None:
                break
            time.sleep(.05)
        assert r.birth(pid) is None
        log = r.jobdir(tmp_path,jid)/'worker.log'
        assert log.stat().st_size == 16384
        assert log.read_text().startswith('delayed diagnostic')
        capture = r.read(tmp_path,r.jobdir(tmp_path,jid)/'log_capture.json')
        assert capture['truncated'] and capture['retained_bytes']==16384
        assert capture['received_bytes'] > capture['limit_bytes']
        status = r.research_status(tmp_path,jid)
        assert status['status']=='failed' and status['successful_completed']==0
        assert status['attempt_number']==1 and status['queue']==0
    finally:
        if parent.poll() is None:
            parent.communicate('\n', timeout=5)
        for stream in (parent.stdin,parent.stdout,parent.stderr):
            stream.close()


def test_chunked_teacher_matches_actual_full_forward(tmp_path, monkeypatch):
    import torch
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False)])
            self.layers[0].mlp = torch.nn.Module()
            self.layers[0].mlp.down_proj = torch.nn.Linear(3, 3, bias=False)
            self.calls = 0
            with torch.no_grad():
                self.layers[0].weight.copy_(torch.eye(3))
                self.layers[0].mlp.down_proj.weight.copy_(2*torch.eye(3))
        def forward(self, input_ids, **kw):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            self.calls += 1
            x = torch.nn.functional.one_hot(input_ids % 3, 3).float()
            # Exercise actual residual and MLP observation hooks.
            x = self.layers[0].mlp.down_proj(x)
            return SimpleNamespace(last_hidden_state=self.layers[0](x))
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.head = torch.nn.Linear(3, 7, bias=False)
            self.config = SimpleNamespace(vocab_size=7, hidden_size=3, model_type='qwen3_5_text')
            self.chunks = []
            with torch.no_grad():
                self.head.weight.copy_(torch.arange(21).reshape(7,3)/10)
            self.eval().requires_grad_(False)
        @property
        def base_model(self): return self.model
        def get_output_embeddings(self): return self.head
        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.model(input_ids).last_hidden_state))
    model = LM()
    row = {'id': 'long', 'tokens': [0, 1, 2, 3, 4, 5, 6]*40}
    with torch.inference_mode():
        batch = torch.tensor([row['tokens']])
        logits = model(batch[:,:-1]).logits[0]
        expected = logits.log_softmax(-1).gather(1,batch[0,1:,None]).squeeze(1).numpy()
        accuracy = float((logits.argmax(-1)==batch[0,1:]).float().mean())
    model.model.calls=0
    handle = model.head.register_forward_pre_hook(lambda m,args: model.chunks.append(args[0].shape[1]))
    monkeypatch.setattr(w, 'TEACHER_HEAD_TOKENS', 13)
    artifacts = w.Artifacts(tmp_path,tmp_path,2**20)
    obs = w.Observations(model,{'layers':[0],'mlp_layers':[0],'positions':{'long':[0,21,278]},'mean_answer_tokens':12},artifacts,'original')
    try:
        metrics,calls = w.teacher(model,[row],artifacts,'original',obs)
    finally:
        obs.close(); handle.remove()
    actual = next(a for a in artifacts.manifest if a['kind']=='target_logprobs')
    np.testing.assert_allclose(np.load(tmp_path/actual['path']),expected,atol=5e-7)
    assert metrics['nll']==pytest.approx(-expected.mean(),abs=1e-6)
    assert metrics['accuracy']==pytest.approx(accuracy)
    assert model.model.calls==1 and max(model.chunks)==13
    assert sum(model.chunks)==len(row['tokens'])-1
    assert calls==1+len(model.chunks)
    assert metrics['execution']['backbone_forward']==1
    assert metrics['execution']['causal_lm_forward']==0
    assert metrics['execution']['output_head_forward']==len(model.chunks)
    assert obs.called['model.layers.0']['calls']==1
    assert obs.called['model.layers.0.mlp.down_proj']['calls']==1
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[0].mlp.down_proj._forward_pre_hooks


def test_prompt_suffix_exact_and_teacher_untouched(tmp_path, monkeypatch):
    suffix = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
    world(tmp_path,monkeypatch,prompt_suffix=suffix)
    assert r.config(tmp_path)['prompt_suffix']==suffix
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            assert messages==[{'role':'user','content':'question'+suffix}]
            assert kw['enable_thinking']
            return [1,2]
        def encode(self,text,**kw):
            assert text=='exact reference'
            return [2,3]
    tok=Tokenizer()
    assert w.tokenize(tok,[{'id':'a','prompt':'question'}],'generate',suffix)[0]['tokens']==[1,2]
    assert w.tokenize(tok,[{'id':'a','reference':'exact reference'}],'teacher_forced',suffix)[0]['tokens']==[2,3]
    assert w.tokenize(tok,[{'id':'a','tokens':[3,4]}],'teacher_forced',suffix)[0]['tokens']==[3,4]
    c=r.read(tmp_path,'research.json'); c['prompt_suffix']=None
    r.atomic(tmp_path,'research.json',c)
    with pytest.raises(ValueError,match='prompt_suffix'): r.config(tmp_path)


def test_optional_symbolic_grading():
    pytest.importorskip('math_verify')
    assert w.grade(r'\boxed{\sqrt{4}}','2')[1] is True
    assert w.grade(r'\boxed{\frac{2}{4}}',r'\frac{1}{2}')[1] is True
    assert w.grade(r'\boxed{\sqrt{5}}','2')[1] is False
    assert w.grade(r'\boxed{2} then \boxed{\sqrt{4}', '2')[1] is None
    assert w.grade('unfinished reasoning 2','2')[1] is None


def test_chunked_teacher_head_error_removes_observation_hooks(tmp_path, monkeypatch):
    import torch
    _, _, _ = world(tmp_path,monkeypatch)
    model = w.inference.load_inference_model(tmp_path,'model')
    model.config.model_type='qwen3_5_text'
    model.base_model=model.model
    def backbone(input_ids, **kw):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        x=torch.nn.functional.one_hot(input_ids % 3,3).float()
        return SimpleNamespace(last_hidden_state=model.model.layers[0](x))
    model.model.forward=backbone
    class FailingHead(torch.nn.Module):
        def forward(self,x): raise RuntimeError('head control failure')
    model.get_output_embeddings=lambda: FailingHead()
    monkeypatch.setattr(w.inference,'load_inference_model',lambda *a,**kw:model)
    request=dict(type='teacher_forced',model='original',split='discovery',ids=['b'],observations={'layers':[0],'mlp_layers':[0]})
    jid=reserve(tmp_path,monkeypatch,request)
    with pytest.raises(RuntimeError,match='head control failure'):
        w.execute(tmp_path,jid)
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[0].mlp.down_proj._forward_pre_hooks
    assert not (r.jobdir(tmp_path,jid)/'result.json').exists()


def test_generation_mixed_eos_presence_and_observation_counts(tmp_path,monkeypatch):
    import torch
    c, _, tokenizer=world(tmp_path,monkeypatch)
    model=w.inference.load_inference_model(tmp_path,'model')
    artifacts=w.Artifacts(tmp_path,tmp_path,2**20)
    obs=w.Observations(model,{'layers':[0],'mean_answer_tokens':3,
                            'generation_logits': {'token_ids': [1, 0], 'positions': {'a': [0, 1], 'b': [1, 2]}}},
                       artifacts,'original')
    seen_masks=[]
    def sample(logits,seen,random_values,sampling):
        seen_masks.append(seen.clone())
        return torch.tensor([4,0] if len(seen_masks)==1 else [1,4])
    monkeypatch.setattr(w,'sample_tokens',sample)
    try:
        rows,metrics,calls=w.generate(model,tokenizer,[{'id':'a','tokens':[1]},{'id':'b','tokens':[2,1]}],c,spec(),obs)
    finally:
        obs.close()
    assert [row['continuation_ids'] for row in rows]==[[4],[0,4]]
    assert all(row['finish_reason']=='eos' for row in rows)
    assert metrics['generated_tokens']==3 and metrics['cutoff']==0
    assert not seen_masks[0].any()  # prompt tokens do not incur presence penalty
    assert seen_masks[1].tolist()==[[False,False,False,False,True],[True,False,False,False,False]]
    assert calls==3  # prefill + generated observation, including EOS token
    means=[a for a in artifacts.manifest if a['kind']=='observation_mean']
    assert [(a['id'],a['mean_token_count']) for a in means]==[('a',1),('b',2)]
    np.testing.assert_array_equal(np.load(tmp_path/means[0]['path']),[0.,2.,0.])
    assert model.seen_calls[2][1].tolist()==[[0,1,1,0],[1,1,1,1]]
    scores = next(a for a in artifacts.manifest if a['kind'] == 'generation_logits')
    assert scores['token_ids'] == [1, 0]
    assert json.loads((tmp_path / scores['path']).read_text()) == [
        {'id': 'a', 'sample': 0, 'continuation_offset': 0, 'chosen_token_id': 4, 'logits': [2.0, 0.0]},
        {'id': 'b', 'sample': 0, 'continuation_offset': 1, 'chosen_token_id': 4, 'logits': [0.0, 2.0]},
    ]


def test_first_token_runtime_bias_preserves_raw_logits_weights_and_later_steps(tmp_path, monkeypatch):
    import torch
    c, _, tokenizer = world(tmp_path, monkeypatch)
    c['batch_size'] = 1
    c['sampling'].update(top_k=1, temperature=1., presence_penalty=0.)
    model = w.inference.load_inference_model(tmp_path, 'model')
    before = {name: value.clone() for name, value in model.state_dict().items()}

    def forward(input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        # Fixed two-step fixture, never trained. Persisting the bias would prevent EOS.
        values = [0., 2., 1., 0., -1.] if past_key_values is None else [0., 0., 0., 0., 1.]
        logits = torch.tensor(values).view(1, 1, 5).expand(input_ids.shape[0], input_ids.shape[1], 5)
        return SimpleNamespace(logits=logits, past_key_values=1)

    model.forward = forward
    artifacts = w.Artifacts(tmp_path, tmp_path, 2**20)
    obs = w.Observations(model, {'first_token_ids': [0, 1]}, artifacts, 'original')
    request = {'n': 1, 'max_new_tokens': 2,
               'first_token_logit_bias': [{'token_id': 0, 'bias': 10.}]}
    rows, _, _ = w.generate(model, tokenizer, [{'id': 'a', 'tokens': [1]}], c, request, obs)
    obs.close()
    assert rows[0]['continuation_ids'] == [0, 4]
    assert rows[0]['finish_reason'] == 'eos'
    raw = next(a for a in artifacts.manifest if a['kind'] == 'first_token_logits')
    assert json.loads((tmp_path / raw['path']).read_text())[0]['logits'] == [0., 2.]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    plain = w.Observations(model, {}, None, 'original')
    baseline, _, _ = w.generate(model, tokenizer, [{'id': 'a', 'tokens': [1]}], c,
                                 {'n': 1, 'max_new_tokens': 2}, plain)
    plain.close()
    assert baseline[0]['continuation_ids'] == [1, 4]


def test_first_token_runtime_bias_rejects_vllm_and_out_of_vocabulary(tmp_path, monkeypatch):
    c, _, _ = world(tmp_path, monkeypatch)
    request = spec(first_token_logit_bias=[{'token_id': 5, 'bias': .5}])
    model = w.inference.load_inference_model(tmp_path, 'model')
    with pytest.raises(ValueError, match='outside vocabulary'):
        w.validate_tokens(model, [{'id': 'a', 'tokens': [1]}], c, request)
    c['generation_backend'] = 'vllm'
    with pytest.raises(ValueError, match='Transformers generation'):
        r.validate_spec(tmp_path, c, spec(first_token_logit_bias=[{'token_id': 1, 'bias': .5}]))


def test_paired_accuracy_includes_unparsed_failures():
    original=[{'id':'a','sample':0,'grade':None},{'id':'b','sample':0,'grade':True},{'id':'c','sample':0,'grade':False}]
    edited=[{'id':'a','sample':0,'grade':True},{'id':'b','sample':0,'grade':None},{'id':'c','sample':0,'grade':True}]
    counts=w.paired_accuracy_counts(original,edited)
    assert counts == {'wrong_to_right':2,'right_to_wrong':1,'unparsed_to_correct':1,'correct_to_unparsed':1}
    assert counts['wrong_to_right']-counts['right_to_wrong'] == sum(r['grade'] is True for r in edited)-sum(r['grade'] is True for r in original)


def test_inference_cast_preserves_source_fp32_decay_and_rotary_buffer():
    import torch
    model=torch.nn.Module()
    model.writer=torch.nn.Linear(2,2,bias=False)
    model.decay=torch.nn.Parameter(torch.tensor([.123456789]))
    model.register_buffer('inv_freq',torch.tensor([.123456789,.000000123456789]))
    before_param=model.decay.detach().clone();before_buffer=model.inv_freq.clone()
    w.cast_for_inference(model,'cpu',torch.bfloat16,{'decay':'torch.float32','writer.weight':'torch.bfloat16'})
    assert model.writer.weight.dtype==torch.bfloat16
    assert model.decay.dtype==model.inv_freq.dtype==torch.float32
    assert torch.equal(model.decay,before_param) and torch.equal(model.inv_freq,before_buffer)
    assert not model.decay.requires_grad
