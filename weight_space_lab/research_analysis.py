"""Offline affine evidence from verified forward artifacts; never fits or chooses weights."""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from . import core, evaluation as ev


def research_export_tensor(root, checkpoint, tensor, output):
    """Save a bounded exact parameter array for CPU analysis, never an edited checkpoint."""
    root=Path(root)
    source=core.local(root,checkpoint)
    source_hash=core.sha256(source)
    with safe_open(source,framework='np') as handle:
        shape=handle.get_slice(tensor).get_shape()
        if int(np.prod(shape)) > 16_777_216:
            raise ValueError('parameter export exceeds16Mentries; select smaller tensor or explicit slices')
        array=handle.get_tensor(tensor)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('finite numeric parameter required')
    target=core.local(root,output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        np.save(stream,array,allow_pickle=False)
    if core.sha256(source)!=source_hash:
        target.unlink()
        raise ValueError('source changed during export')
    result={'array_path':output,'array_sha256':core.sha256(target),'source_sha256':source_hash,
            'tensor':tensor,'shape':list(array.shape),'dtype':str(array.dtype),'model_forward_calls':0,
            'interpretation':'Exact stored axes; inspect actual module equation before algebra. No edit or fitting.'}
    ev.write_report(root,result,output+'.json')
    return result


def _job(root, job_id):
    from . import research
    result = research.research_result(root, job_id)
    if result.get('status') == 'failed':
        raise ValueError('successful observed forward required')
    snapshot = research.load_snapshot(root, job_id)
    artifacts = list(result.get('artifacts', []))
    if result.get('artifact_manifest'):
        artifacts += research.read(root, result['artifact_manifest']['path'])
    return result, snapshot, list({a['path']: a for a in artifacts}.values())


def _matrix(root, checkpoint, tensor):
    with safe_open(core.local(root, checkpoint), framework='np') as h:
        a = h.get_tensor(tensor)
    if a.ndim != 2 or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError('finite floating matrix required')
    return a


def _effective(array, dtype):
    if dtype == 'float32':
        return np.asarray(array, dtype=np.float32)
    if dtype != 'bfloat16':
        raise ValueError('unsupported recorded inference dtype')
    import torch
    # Dtype conversion of weights only; no network, derivative or optimization.
    return torch.from_numpy(np.array(array, copy=True)).to(torch.bfloat16).float().numpy()


def research_prepare_edit(root, plan, tensor, rows, columns, contexts, direction, basis, output):
    """Bind an explicit plan to real recorded affine inputs, including full Qwen output width.

    The caller chooses the plan, coordinates, contexts, direction and local bound.
    Readiness means arithmetic/evidence fields suffice for a causal probe only.
    """
    root = Path(root)
    source = core.local(root, plan['checkpoint'])
    source_hash = core.sha256(source)
    if source_hash != plan['source_sha256']:
        raise ValueError('source hash mismatch')
    replacements, numeric = core.prepare(root, plan)
    original = _matrix(root, plan['checkpoint'], tensor)
    if tensor not in replacements:
        raise ValueError('plan must include analyzed tensor')
    core._indices(rows, original.shape[0]); core._indices(columns, original.shape[1])
    if len(rows) > 4096 or len(columns) > 256 or len(rows)*len(columns) > 262144:
        raise ValueError('bounded selected matrix required')
    if not isinstance(contexts, list) or not 2 <= len(contexts) <= 64:
        raise ValueError('select2..64 actual reference/failure contexts')
    u = np.asarray(direction, dtype=np.float64)
    if u.shape != (len(rows),) or not np.isfinite(u).all() or not np.linalg.norm(u):
        raise ValueError('explicit finite nonzero output direction required')
    u /= np.linalg.norm(u)
    cache, weights_cache, observations, seen = {}, {}, [], set()
    uncovered = False
    for c in contexts:
        role = c['role']
        if role not in ('reference', 'failure'):
            raise ValueError('explicit reference/failure role required')
        jid, path = c['job_id'], c['path']
        if jid not in cache:
            cache[jid] = _job(root, jid)
        result, snapshot, artifacts = cache[jid]
        model_file = str(Path(snapshot['config']['model_dir']) / ev.MODEL_FILE)
        if result['provenance'].get(model_file) != source_hash:
            raise ValueError('observed source differs from plan source')
        matching = [a for a in artifacts if a['path'] == path and a.get('kind') in ('observations', 'observation_mean')]
        if len(matching) != 1:
            raise ValueError('unique bound observation artifact required')
        a = matching[0]
        if a.get('module') + '.weight' != tensor:
            raise ValueError('artifact must be actual input of the selected affine writer')
        p = core.local(root, path)
        if core.sha256(p) != a['sha256']:
            raise ValueError('observation artifact hash mismatch')
        x = np.load(p, mmap_mode='r', allow_pickle=False)
        index = c.get('row_index', 0)
        try:
            if x.ndim == 1:
                if index != 0:
                    raise ValueError('mean artifact has one vector')
                vector = np.array(x, dtype=np.float64)
            elif x.ndim == 2 and type(index) is int and 0 <= index < x.shape[0]:
                vector = np.array(x[index], dtype=np.float64)
            else:
                raise ValueError('explicit observed vector index required')
        finally:
            x._mmap.close()
        if vector.shape != (original.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('observed input width/nonfinite mismatch')
        key = (jid, path, index)
        if key in seen:
            raise ValueError('duplicate context vector cannot inflate evidence')
        seen.add(key)
        if a['model'] not in ('original', 'edited'):
            raise ValueError('unknown observed model')
        candidate = snapshot['spec'].get('candidate_dir') if a['model'] == 'edited' else None
        if a['model'] == 'edited' and not candidate:
            raise ValueError('edited observation lacks candidate provenance')
        dtype = snapshot['config']['inference_dtype']
        key_weights = (candidate, dtype)
        if key_weights not in weights_cache:
            current = original
            if candidate:
                _, patch = ev.verify_candidate(root, snapshot['config']['model_dir'], candidate)
                current = patch.get(tensor, original)
            current, proposed = _effective(current, dtype), _effective(replacements[tensor], dtype)
            old = current[np.ix_(rows, columns)].astype(np.float64)
            new = proposed[np.ix_(rows, columns)].astype(np.float64)
            complete = np.count_nonzero(current != proposed) == np.count_nonzero(old != new)
            weights_cache[key_weights] = old, new, complete
        old, new, complete = weights_cache[key_weights]
        uncovered |= not complete
        input_values = vector[columns]
        contribution, delta = old @ input_values, (new-old) @ input_values
        observations.append({'role': role, 'job_id': jid, 'artifact_path': path, 'artifact_sha256': a['sha256'],
            'id': a['id'], 'sample': a.get('sample', 0), 'row_index': index, 'module': a['module'],
            'recorded_positions': a.get('positions'), 'mean_token_count': a.get('mean_token_count'),
            'observed_model': a['model'], 'inference_dtype': dtype,
            'signed_selected_input': input_values.tolist(), 'current_selected_contribution': contribution.tolist(),
            'fixed_input_selected_delta': delta.tolist(), 'delta_l2': float(np.linalg.norm(delta)),
            'delta_projection': float(u @ delta)})
    reference = [o for o in observations if o['role'] == 'reference']
    failure = [o for o in observations if o['role'] == 'failure']
    if not reference or not failure:
        raise ValueError('both role groups required')
    reasons = []
    if uncovered:
        reasons.append('analyzed block omits incremental changes in this tensor')
    fields = ('failure_description', 'structural_role', 'mechanism_hypothesis', 'pairing_justification',
              'value_derivation', 'alternative_explanation', 'expected_behavior', 'disconfirming_result')
    for key in fields:
        if not isinstance(basis.get(key), str) or not basis[key].strip():
            reasons.append('missing rationale: ' + key)
    controls = basis.get('controls', [])
    if not isinstance(controls, list) or not controls or any(not isinstance(x, dict) or not x.get('kind') or not x.get('purpose') for x in controls):
        reasons.append('explicit causal control plan required')
    bound = basis.get('max_reference_output_delta')
    maximum = max(o['delta_l2'] for o in reference)
    if type(bound) not in (int, float) or not np.isfinite(bound) or bound < 0 or maximum > bound:
        reasons.append('caller reference-output bound absent or exceeded')
    projection = float(np.mean([o['delta_projection'] for o in failure]))
    if projection <= 0:
        reasons.append('declared desired failure direction not supported locally')
    if core.sha256(source) != source_hash:
        raise ValueError('source changed during evidence analysis')
    result = {'kind': 'research_affine_edit_evidence', 'schema_version': 1,
        'source_sha256': source_hash, 'plan': plan, 'numeric_preview': numeric,
        'tensor': tensor, 'layout': 'out,in', 'rows': rows, 'columns': columns,
        'desired_output_direction_unit': u.tolist(), 'basis': basis, 'observations': observations,
        'distinct_problem_ids': len({o['id'] for o in observations}),
        'role_problem_counts': {role: len({o['id'] for o in observations if o['role'] == role}) for role in ('reference','failure')},
        'max_reference_delta_l2': maximum, 'mean_failure_delta_projection': projection,
        'decision': 'defer' if reasons else 'ready_for_causal_probe', 'reasons': reasons,
        'model_forward_calls': 0, 'candidate_created': False,
        'interpretation': 'Actual artifact provenance and explicit affine arithmetic only. Means are pooled inputs, not full trajectories; caller labels and independence are unverified. Cast weights model the recorded inference dtype, but downstream nonlinear effects/output rounding and changed future inputs require actual probes. Readiness never proves a causal role, preservation or performance.'}
    ev.write_report(root, result, output)
    return {'analysis_path': output, 'analysis_sha256': core.sha256(core.local(root, output)),
            'decision': result['decision'], 'reasons': reasons, 'distinct_problem_ids': result['distinct_problem_ids'],
            'max_reference_delta_l2': maximum, 'mean_failure_delta_projection': projection,
            'model_forward_calls': 0, 'candidate_created': False}


def research_make_contexts(root, job_id, ids, output, which='original', max_tokens=8192):
    """Materialize actual generated prefixes as teacher-forced discovery contexts, never labels.

    Source example IDs/split are retained so the research forward tool enforces
    development provenance. Context selection and failure interpretation are LLM decisions.
    """
    root = Path(root)
    if type(max_tokens) is not int or not 2 <= max_tokens <= 32768:
        raise ValueError('context length must be2..32768')
    if not isinstance(ids, list) or not 1 <= len(ids) <= 128 or len(set(ids)) != len(ids):
        raise ValueError('select1..128 unique source IDs')
    result, snapshot, artifacts = _job(root, job_id)
    if snapshot['spec']['type'] != 'generate' or which not in ('original','edited'):
        raise ValueError('actual generation provenance required')
    a = [a for a in artifacts if a.get('kind') == 'generations' and a.get('model') == which]
    if len(a) != 1:
        raise ValueError('unique actual generation artifact required')
    p = core.local(root, a[0]['path'])
    if core.sha256(p) != a[0]['sha256']:
        raise ValueError('generation artifact hash mismatch')
    generated = json.loads(p.read_text())
    by_id = {r['id']: r for r in generated if r.get('sample', 0) == 0}
    if any(i not in by_id for i in ids):
        raise ValueError('requested source ID absent')
    rows = []
    for i in ids:
        r = by_id[i]
        tokens = (r['input_ids'] + r['continuation_ids'])[:max_tokens]
        if len(tokens) < 2:
            raise ValueError('insufficient actual context tokens')
        rows.append({'id': i, 'tokens': tokens})
    split = snapshot['spec']['split']
    data = {'name': 'actual-generated-prefix-contexts', 'version': '1', 'splits': {split: rows},
            'provenance': {'job_id': job_id, 'model': which, 'generation_sha256': a[0]['sha256'],
                           'max_tokens': max_tokens, 'sample': 0},
            'labels_status': 'actual recorded prefixes only; no semantic failure labels inferred'}
    ev.write_report(root, data, output)
    return {'dataset_path': output, 'sha256': core.sha256(core.local(root, output)), 'examples': len(rows),
            'input_tokens': sum(len(r['tokens']) for r in rows), 'model_forward_calls': 0}
