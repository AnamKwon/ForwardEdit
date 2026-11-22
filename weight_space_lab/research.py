"""Source-bound serial research jobs. No model execution or edit selection here."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid
import numpy as np
from . import core, evaluation as ev


def read(root, path):
    return json.loads(core.local(Path(root), str(path)).read_text())


def atomic(root, path, value):
    target = core.local(Path(root), str(path))
    temp = core.local(Path(root), str(target) + '.' + uuid.uuid4().hex)
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, allow_nan=False, separators=(',', ':'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be integer in [{low}, {high}]')
    return value


def config(root):
    c = read(root, 'research.json')
    for key, limit in [('successful_forward_budget', 50), ('attempt_budget', 75),
                       ('max_items', 128), ('max_new_tokens', 32768), ('max_input_tokens', 32768),
                       ('max_total_input_tokens', 524288)]:
        integer(c[key], 1, limit, key)
    if c['device'] not in ('cuda', 'cpu') or c['inference_dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('invalid inference device/dtype')
    if c.get('generation_backend', 'transformers') not in ('transformers','vllm'):
        raise ValueError('unsupported generation backend')
    if type(c.get('native_precision_load', False)) is not bool:
        raise ValueError('native_precision_load must be boolean')
    if c.get('native_precision_load') and (c['inference_dtype'] != 'bfloat16'
            or c.get('generation_backend', 'transformers') != 'transformers' or c.get('multimodal')):
        raise ValueError('native_precision_load requires text-only BF16 Transformers inference')
    if c.get('generation_backend') == 'vllm':
        if c['device'] != 'cuda' or c['inference_dtype'] != 'bfloat16' or not Path(c.get('vllm_python','')).is_file():
            raise ValueError('vLLM requires pinned executable, CUDA and bfloat16')
    for k in ('max_artifact_bytes_per_job', 'max_artifact_bytes_total'):
        integer(c[k], 1024, 2**50, k)
    integer(c.get('batch_size', 8), 1, 64, 'batch_size')
    if 'prompt_suffix' in c and (not isinstance(c['prompt_suffix'], str) or len(c['prompt_suffix']) > 16384):
        raise ValueError('prompt_suffix must be a string of at most 16384 characters')
    if 'multimodal' in c:
        from .multimodal import validate_settings
        validate_settings(root, c['multimodal'])
        if c.get('generation_backend', 'transformers') != 'transformers':
            raise ValueError('native multimodal inputs require the Transformers backend')
    s = c['sampling']
    integer(s['seed'], 0, 2**63-1, 'seed')
    integer(s['top_k'], 1, 20, 'top_k')
    for key, lo, hi in [('temperature', 0, 100), ('top_p', 0, 1), ('presence_penalty', -100, 100)]:
        v = s[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi or (key != 'presence_penalty' and v == 0):
            raise ValueError('invalid sampling ' + key)
    return c


def dataset(root, path):
    data = read(root, path)
    splits = data.get('splits', {})
    if not splits or any(k not in ('discovery', 'validation') for k in splits):
        raise ValueError('research datasets permit discovery/validation only')
    seen = set()
    input_owners, group_owners, image_hashes = {}, {}, {}
    for split, rows in splits.items():
        if not isinstance(rows, list):
            raise ValueError('split must be a list')
        for row in rows:
            i = row.get('id')
            if not isinstance(i, str) or not i or len(i) > 256 or i in seen:
                raise ValueError('unique bounded string IDs required')
            seen.add(i)
            if 'tokens' in row:
                t = row['tokens']
                if not isinstance(t, list) or len(t) < 2 or any(type(v) is not int or v < 0 for v in t):
                    raise ValueError('tokens must be >=2 nonnegative integers')
            elif not isinstance(row.get('prompt'), str) and not isinstance(row.get('reference'), str):
                raise ValueError('prompt, reference or tokens required')
            if 'images' in row:
                if (not isinstance(row['images'], list) or len(row['images']) > 4
                        or any(not isinstance(name, str) for name in row['images'])):
                    raise ValueError('images must list up to four local image paths')
                for name in row['images']:
                    if not core.local(Path(root), name).is_file():
                        raise ValueError('missing local image: ' + name)
            # Different row IDs or labels must not disguise the same input as
            # independent validation. Image identity uses bytes, not filenames.
            images = []
            for name in row.get('images', []):
                path = core.local(Path(root), name)
                if path not in image_hashes:
                    image_hashes[path] = core.sha256(path)
                images.append(image_hashes[path])
            images = tuple(images)
            keys = []
            if 'tokens' in row:
                keys.append(('tokens', tuple(row['tokens']), images))
            if isinstance(row.get('prompt'), str):
                keys.append(('prompt', row['prompt'].replace('\r\n', '\n').strip(), images))
            elif 'tokens' not in row and isinstance(row.get('reference'), str):
                keys.append(('reference', row['reference'].replace('\r\n', '\n').strip(), images))
            for key in keys:
                previous = input_owners.setdefault(key, split)
                if previous != split:
                    raise ValueError('cross-split input overlap between ' + previous + ' and ' + split)
            if 'group_id' in row:
                group = row['group_id']
                if not isinstance(group, str) or not group.strip() or len(group) > 256:
                    raise ValueError('group_id must be a nonempty bounded string')
                previous = group_owners.setdefault(group.strip(), split)
                if previous != split:
                    raise ValueError('cross-split group overlap between ' + previous + ' and ' + split)
    return splits


def snapshot(root, c, spec):
    directory = ev._model_dir(root, c['model_dir'])
    paths = {str(directory / 'model.safetensors'): c['source_sha256'],
             str(directory / 'config.json'): c['config_sha256'],
             c['dataset_path']: c['dataset_sha256']}
    for name,digest in c.get('model_static_sha256',{}).items():
        paths[str(core.local(root,str(directory/name)))]=digest
    # Bind tokenizer/chat-template files too; reject nested escaping symlinks.
    for p in directory.rglob('*'):
        q = core.local(root, str(p))
        if q.is_file():
            paths.setdefault(str(q), core.sha256(q))
    paths['research.json'] = core.sha256(core.local(root, 'research.json'))
    if spec.get('dataset_path'):
        p = core.local(root, spec['dataset_path'])
        paths[str(p)] = core.sha256(p)
    data = dataset(root, spec.get('dataset_path', c['dataset_path']))
    for row in data[spec['split']]:
        if row['id'] in spec['ids']:
            for name in row.get('images', []):
                image = core.local(root, name)
                paths[str(image)] = core.sha256(image)
    if c.get('multimodal'):
        for key in ('vision_checkpoint', 'config'):
            p = core.local(root, c['multimodal'][key])
            paths[str(p)] = core.sha256(p)
        for p in core.local(root, c['multimodal']['processor_dir']).rglob('*'):
            if p.is_file():
                paths[str(core.local(root, str(p)))] = core.sha256(p)
    if spec.get('candidate_dir'):
        ev.verify_candidate(root, c['model_dir'], spec['candidate_dir'])
        for name in ('manifest.json', 'replacements.safetensors'):
            p = core.local(root, str(Path(spec['candidate_dir']) / name))
            paths[str(p)] = core.sha256(p)
    result = {str(core.local(root, k).relative_to(root)): v for k, v in paths.items()}
    verify_hashes(root, result)
    return result


def verify_hashes(root, hashes):
    for p, digest in hashes.items():
        if core.sha256(core.local(root, p)) != digest:
            raise ValueError('provenance hash mismatch: ' + p)


def validate_spec(root, c, spec):
    spec = json.loads(json.dumps(spec, allow_nan=False))
    if set(spec) - {'type', 'model', 'candidate_dir', 'split', 'ids', 'n', 'max_new_tokens', 'observations', 'dataset_path', 'first_token_logit_bias'}:
        raise ValueError('unknown research spec field')
    if spec.get('type') not in ('generate', 'teacher_forced') or spec.get('model') not in ('original', 'edited', 'both'):
        raise ValueError('invalid research type/model')
    if spec['model'] == 'both' and spec['type'] != 'generate':
        raise ValueError('both requires generation')
    if spec['model'] != 'original' and not spec.get('candidate_dir'):
        raise ValueError('candidate_dir required')
    if spec.get('split') not in ('discovery', 'validation'):
        raise ValueError('research permits discovery/validation only')
    base = dataset(root, c['dataset_path'])
    data = dataset(root, spec.get('dataset_path', c['dataset_path']))
    ids = spec.get('ids')
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('exact unique IDs required')
    integer(len(ids), 1, c['max_items'], 'items')
    split = spec['split']
    provenance = {r['id'] for r in base.get(split, [])}
    by_id = {r['id']: r for r in data.get(split, [])}
    if not set(ids) <= provenance or not set(ids) <= by_id.keys():
        raise ValueError('IDs must retain original development split provenance')
    if spec['type'] == 'generate' and any('tokens' not in by_id[i] and not isinstance(by_id[i].get('prompt'), str) for i in ids):
        raise ValueError('generation requires prompt or tokens')
    if any(by_id[i].get('images') for i in ids):
        if not c.get('multimodal') or spec['type'] != 'generate':
            raise ValueError('image inputs require configured native multimodal generation')
        if any(by_id[i].get('images') and not isinstance(by_id[i].get('prompt'), str) for i in ids):
            raise ValueError('image inputs require their original prompt text')
    spec['n'] = integer(spec.get('n', 1), 1, 4, 'n')
    if spec['type'] == 'teacher_forced' and spec['n'] != 1:
        raise ValueError('teacher_forced n must be 1')
    if spec['type'] == 'generate':
        integer(spec.get('max_new_tokens'), 1, c['max_new_tokens'], 'max_new_tokens')
    if 'first_token_logit_bias' in spec:
        bias = spec['first_token_logit_bias']
        if (spec['type'] != 'generate' or c.get('generation_backend', 'transformers') != 'transformers'
                or not isinstance(bias, list) or not 1 <= len(bias) <= 32):
            raise ValueError('first_token_logit_bias requires Transformers generation and 1..32 entries')
        tokens = []
        for entry in bias:
            if (not isinstance(entry, dict) or set(entry) != {'token_id', 'bias'}
                    or type(entry['token_id']) is not int or entry['token_id'] < 0
                    or type(entry['bias']) not in (int, float)
                    or not math.isfinite(entry['bias']) or abs(entry['bias']) > 100):
                raise ValueError('first_token_logit_bias requires nonnegative token IDs and finite biases in [-100,100]')
            tokens.append(entry['token_id'])
        if len(tokens) != len(set(tokens)):
            raise ValueError('first_token_logit_bias requires unique token IDs')
    obs = spec.get('observations', {})
    if c.get('generation_backend') == 'vllm' and spec['type'] == 'generate' and obs:
        raise ValueError('vLLM generation observations require a separate actual teacher-forced context pass')
    if set(obs) - {'layers', 'mlp_layers', 'module_inputs', 'positions', 'mean_answer_tokens', 'first_token_ids', 'generation_logits'}:
        raise ValueError('unknown observation field')
    module_inputs = obs.get('module_inputs', [])
    if (not isinstance(module_inputs, list) or len(module_inputs) > 8
            or any(not isinstance(name, str) or not name for name in module_inputs)
            or len(set(module_inputs)) != len(module_inputs)):
        raise ValueError('module_inputs requires up to eight unique module names')
    if 'first_token_ids' in obs:
        token_ids = obs['first_token_ids']
        if (spec['type'] != 'generate' or not isinstance(token_ids, list)
                or not 1 <= len(token_ids) <= 32
                or any(type(token) is not int or token < 0 for token in token_ids)
                or len(set(token_ids)) != len(token_ids)):
            raise ValueError('first_token_ids requires generation and 1..32 unique nonnegative IDs')
    if 'generation_logits' in obs:
        capture = obs['generation_logits']
        if (spec['type'] != 'generate' or not isinstance(capture, dict)
                or set(capture) != {'token_ids', 'positions'}):
            raise ValueError('generation_logits requires generation, token_ids and positions')
        token_ids, positions = capture['token_ids'], capture['positions']
        if (not isinstance(token_ids, list) or not 1 <= len(token_ids) <= 32
                or any(type(token) is not int or token < 0 for token in token_ids)
                or len(set(token_ids)) != len(token_ids)):
            raise ValueError('generation_logits requires 1..32 unique nonnegative token IDs')
        if not isinstance(positions, dict) or not positions or not positions.keys() <= set(ids):
            raise ValueError('generation_logits positions require selected IDs')
        for offsets in positions.values():
            if (not isinstance(offsets, list) or not 1 <= len(offsets) <= 16
                    or any(type(offset) is not int or not 0 <= offset < spec['max_new_tokens'] for offset in offsets)
                    or len(set(offsets)) != len(offsets)):
                raise ValueError('generation_logits requires 1..16 unique offsets inside the decode limit')
    for key in ('layers', 'mlp_layers'):
        v = obs.get(key, [])
        if not isinstance(v, list) or len(v) > 128 or any(type(i) is not int or i < 0 for i in v) or len(set(v)) != len(v):
            raise ValueError('unique nonnegative layers required')
    integer(obs.get('mean_answer_tokens', 0), 0, 8192, 'mean_answer_tokens')
    positions = obs.get('positions', {})
    if not isinstance(positions, dict) or not positions.keys() <= set(ids):
        raise ValueError('positions require selected IDs')
    for v in positions.values():
        if not isinstance(v, list) or not v or len(v) > 4096 or len(set(v)) != len(v) or any(type(p) is not int or p < 0 for p in v):
            raise ValueError('invalid prediction positions')
    return spec, [by_id[i] for i in ids]


@contextlib.contextmanager
def ledger(root):
    root = Path(root).resolve()
    folder = core.local(root, 'research_jobs')
    folder.mkdir(exist_ok=True)
    with core.local(root, 'research_jobs/ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = sqlite3.connect(core.local(root, 'research_jobs/ledger.sqlite'))
        db.row_factory = sqlite3.Row
        try:
            db.execute('CREATE TABLE IF NOT EXISTS meta (hash TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, pid INT, birth TEXT, error TEXT, bytes INT DEFAULT 0, snapshot_hash TEXT)')
            digest = core.sha256(core.local(root, 'research.json'))
            row = db.execute('SELECT hash FROM meta').fetchone()
            if row and row[0] != digest:
                raise ValueError('immutable research.json changed')
            if not row:
                db.execute('INSERT INTO meta VALUES (?)', (digest,))
            yield db
            db.commit()
        finally:
            db.close()


def birth(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def jobdir(root, job_id):
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('invalid job ID')
    return core.local(Path(root), 'research_jobs/' + job_id)


def load_snapshot(root, jid):
    d = jobdir(root, jid)
    with contextlib.closing(sqlite3.connect(core.local(root, 'research_jobs/ledger.sqlite'))) as db:
        row = db.execute('SELECT snapshot_hash FROM jobs WHERE id=?', (jid,)).fetchone()
    if not row or core.sha256(core.local(root, str(d / 'snapshot.json'))) != row[0]:
        raise ValueError('job snapshot hash mismatch')
    snap = read(root, d / 'snapshot.json')
    if snap['config'] != config(root):
        raise ValueError('snapshot config mismatch')
    return snap


def verified_result(root, jid):
    d = jobdir(root, jid)
    s = load_snapshot(root, jid)
    verify_hashes(root, s['hashes'])
    result = read(root, d / 'result.json')
    observed = result.get('model_forward_invocations')
    engine_only = result.get('execution_counter_scope') == 'engine_submissions_only' and type(result.get('engine_generation_calls')) is int and result['engine_generation_calls'] >= 1 and observed is None
    observed_valid = type(observed) is int and observed >= 1
    if result.get('job_id') != jid or result.get('provenance') != s['hashes'] or not (observed_valid or engine_only) or result.get('successful_samples', 0) < 1:
        raise ValueError('invalid model result')
    if not result.get('artifacts'):
        raise ValueError('missing artifacts')
    artifacts = list(result['artifacts'])
    if result.get('artifact_manifest'):
        manifest = result['artifact_manifest']
        mp = core.local(root, manifest['path'])
        if not mp.is_relative_to(d) or core.sha256(mp) != manifest['sha256']:
            raise ValueError('artifact manifest verification failed')
        artifacts += read(root, mp)
    for a in artifacts:
        p = core.local(root, a['path'])
        if not p.is_relative_to(d) or p.stat().st_size != a['bytes'] or core.sha256(p) != a['sha256']:
            raise ValueError('artifact verification failed')
    if len(json.dumps(result).encode()) > 16384:
        raise ValueError('result summary exceeds 16KiB')
    return result


def reconcile(root, db, explicit=False):
    for row in db.execute("SELECT * FROM jobs WHERE status IN ('pending','running')").fetchall():
        jid = row['id']
        d = jobdir(root, jid)
        try:
            state = read(root, d / 'state.json')
        except (OSError, ValueError):
            state = {}
        status, error = state.get('status'), state.get('error')
        if status == 'succeeded':
            try:
                verified_result(root, jid)
            except (ValueError, OSError, KeyError, TypeError) as e:
                status, error = 'failed', str(e)
        elif status != 'failed':
            if not explicit or (row['pid'] and row['birth'] is not None and birth(row['pid']) == row['birth']):
                continue
            status, error = 'failed', 'worker process ended without verified terminal output'
            atomic(root, d / 'state.json', {'status': status, 'error': error})
        size = sum(core.local(root, str(p)).stat().st_size for p in d.rglob('*') if p.is_file())
        db.execute('UPDATE jobs SET status=?,error=?,bytes=? WHERE id=?', (status, error, size, jid))


def counters(db, c):
    return {'successful_completed': db.execute("SELECT count(*) FROM jobs WHERE status='succeeded'").fetchone()[0],
            'attempt_number': db.execute('SELECT count(*) FROM jobs').fetchone()[0],
            'budget': c['successful_forward_budget']}


def drain_worker(proc, log):
    """Drain all stdio, retaining startup context and final diagnostics in16KiB."""
    received = 0
    retained = min(log.stat().st_size if log.exists() else 0, 16384)
    data=log.read_bytes()[:16384] if log.exists() else b''
    try:
        with log.open('wb') as stream:
            while True:
                block = getattr(proc.stdout,'read1',proc.stdout.read)(8192)
                if not block:
                    break
                received += len(block)
                data+=block
                if len(data)>16384:
                    data=data[:4096]+data[-12288:]
                stream.seek(0);stream.write(data);stream.truncate();stream.flush()
    finally:
        proc.stdout.close()
        proc.wait()
    return {'limit_bytes': 16384, 'received_bytes': received,
            'retained_bytes': len(data), 'truncated': received > 16384-retained}


def research_forward(root, spec):
    root = Path(root).resolve()
    c = config(root)
    spec, _ = validate_spec(root, c, spec)
    hashes = snapshot(root, c, spec)
    with ledger(root) as db:
        reconcile(root, db)
        count = counters(db, c)
        if db.execute("SELECT 1 FROM jobs WHERE status IN ('pending','running')").fetchone():
            raise ValueError('serial queue busy; use research_status to reconcile a dead worker')
        if count['successful_completed'] >= c['successful_forward_budget'] or count['attempt_number'] >= c['attempt_budget']:
            raise ValueError('research budget exhausted')
        used = db.execute('SELECT COALESCE(sum(bytes),0) FROM jobs').fetchone()[0]
        if used + c['max_artifact_bytes_per_job'] > c['max_artifact_bytes_total']:
            raise ValueError('total artifact budget exhausted')
        jid = uuid.uuid4().hex
        d = jobdir(root, jid)
        d.mkdir()
        atomic(root, d / 'snapshot.json', {'spec': spec, 'config': c, 'hashes': hashes})
        atomic(root, d / 'state.json', {'status': 'pending'})
        db.execute('INSERT INTO jobs(id,status,snapshot_hash) VALUES (?,?,?)', (jid, 'pending', core.sha256(core.local(root, str(d / 'snapshot.json')))))
        db.commit()  # durable reservation before spawning
        proc = None
        try:
            log = core.local(root, str(d / 'worker.log'))
            log.touch(exist_ok=False)
            # The detached worker supervises its own bounded log pipe. No pipe
            # endpoint or drain thread belongs to this submitting server.
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'weight_space_lab.research_worker', '--workspace', str(root), '--job', jid], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
            db.execute('UPDATE jobs SET pid=?,birth=? WHERE id=?', (proc.pid, birth(proc.pid), jid))

        except Exception as e:
            if proc is not None:
                proc.terminate()
                proc.wait()
                if proc.stdout is not None:
                    proc.stdout.close()
            atomic(root, d / 'state.json', {'status': 'failed', 'error': str(e)[:1000]})
            db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (str(e)[:1000], jid))
            raise
        return {'job_id': jid, 'status': 'pending', **count, 'attempt_number': count['attempt_number'] + 1, 'job_path': str(d.relative_to(root))}


def research_status(root, job_id):
    directory = jobdir(root, job_id)
    c = config(root)
    with ledger(root) as db:
        reconcile(root, db, explicit=True)
        row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise ValueError('unknown job')
        state = {'job_id': job_id, 'status': row['status'], 'error': row['error'], **counters(db, c), 'queue': db.execute("SELECT count(*) FROM jobs WHERE status IN ('pending','running')").fetchone()[0]}
    try:
        with core.local(Path(root), str(directory / 'progress.json')).open('rb') as stream:
            progress = json.loads(stream.read(1025))
        fields = ('completed_samples', 'total_samples', 'generated_tokens')
        if (isinstance(progress, dict) and progress.get('model') in ('original', 'edited')
                and all(type(progress.get(key)) is int and progress[key] >= 0 for key in fields)
                and progress['completed_samples'] <= progress['total_samples'] <= c['max_items'] * 4
                and progress['generated_tokens'] <= progress['total_samples'] * c['max_new_tokens']):
            state['progress'] = {key: progress[key] for key in ('model', *fields)}
    except (OSError, ValueError):
        pass
    return state


def research_result(root, job_id):
    state = research_status(root, job_id)
    if state['status'] == 'failed':
        return state
    if state['status'] != 'succeeded':
        raise ValueError('job is not terminal')
    return verified_result(root, job_id)


def research_read_array(root, path, indices):
    p = core.local(Path(root), path)
    if not isinstance(indices, list) or not 1 <= len(indices) <= 4096:
        raise ValueError('request 1..4096 coordinate tuples')
    before = core.sha256(p)
    a = np.load(p, mmap_mode='r', allow_pickle=False)
    if not isinstance(a, np.ndarray):
        a.close()
        raise ValueError('npy array required')
    try:
        if a.dtype.kind not in 'fiu':
            raise ValueError('numeric array required')
        entries = []
        for ix in indices:
            if not isinstance(ix, list) or len(ix) != a.ndim or any(type(v) is not int or not 0 <= v < s for v, s in zip(ix, a.shape)):
                raise ValueError('invalid array coordinates')
            value = a[tuple(ix)].item()
            if not math.isfinite(value):
                raise ValueError('nonfinite array entry')
            entries.append({'coordinates': ix, 'value': value})
        result = {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': before, 'entries': entries}
    finally:
        a._mmap.close()
    if core.sha256(p) != before:
        raise ValueError('array changed while reading')
    return result


def research_read_generations(root, job_id, ids, max_chars=300, which='original'):
    integer(max_chars, 1, 2000, 'max_chars')
    if not isinstance(ids, list) or not 1 <= len(ids) <= 128 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or which not in ('original', 'edited'):
        raise ValueError('invalid generation selection')
    result = research_result(root, job_id)
    a = next((a for a in result.get('artifacts', []) if a.get('kind') == 'generations' and a.get('model') == which), None)
    if a is None:
        raise ValueError('no requested generation artifact')
    rows = read(root, a['path'])
    if not set(ids) <= {r['id'] for r in rows}:
        raise ValueError('unknown generation ID')
    return {'rows': [{**{k: r[k] for k in ('id', 'sample', 'generated_tokens', 'input_tokens', 'finish_reason', 'grade')}, 'extracted_answer': (r['extracted_answer'] or '')[:max_chars], 'text': r['text'][:max_chars]} for r in rows if r['id'] in ids]}
