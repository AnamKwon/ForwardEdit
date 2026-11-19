"""Mechanical discovery contexts from observed generation failures; no model or fitting."""
from collections import Counter
import json
from pathlib import Path

from . import core, evaluation as ev


def build_repetition_contexts(root, reference_path, probe_paths, example_ids, output,
                              ngram_size=8, min_occurrences=3, max_positions=4):
    """Pair genuine recurring generated contexts with same-position source references.

    Positions refer to the input token whose next-token prediction completes the
    recurring ngram. The label verifies recurrence, not semantic incoherence.
    """
    root = Path(root)
    if type(ngram_size) is not int or not 2 <= ngram_size <= 32:
        raise ValueError("ngram_size must be2..32")
    if type(min_occurrences) is not int or not 3 <= min_occurrences <= 16:
        raise ValueError("min_occurrences must be3..16")
    if type(max_positions) is not int or not 1 <= max_positions <= 8:
        raise ValueError("max_positions must be1..8")
    if (not isinstance(example_ids, list) or not 1 <= len(example_ids) <= 8
            or any(not isinstance(i, str) for i in example_ids) or len(set(example_ids)) != len(example_ids)):
        raise ValueError("select1..8 unique discovery example ids")
    if not isinstance(probe_paths, list) or not 1 <= len(probe_paths) <= 8 or len(set(probe_paths)) != len(probe_paths):
        raise ValueError("select1..8 distinct probe files")
    ds = ev.load_dataset(root, reference_path)
    refs = {e['id']: e for e in ds['splits'].get('discovery', [])}
    if any(i not in refs for i in example_ids):
        raise ValueError("only existing discovery reference ids allowed")
    probes, provenance, source_hash = {}, [], None
    for path in probe_paths:
        file = core.local(root, path)
        report = json.loads(file.read_text())
        if report.get('split') != 'discovery' or report.get('candidate_dir') is not None:
            raise ValueError("original discovery probes required")
        h = report.get('source_sha256')
        if not isinstance(h, str) or len(h) != 64 or (source_hash is not None and h != source_hash):
            raise ValueError("consistent source hashes required")
        source_hash = h
        eos = report.get('eos_token_ids')
        if not isinstance(eos, list) or any(type(i) is not int or i < 0 for i in eos):
            raise ValueError("explicit EOS ids required")
        for e in report['examples']:
            if e['id'] in probes:
                raise ValueError("duplicate probe id")
            probes[e['id']] = (e, set(eos))
        provenance.append({'path': path, 'sha256': core.sha256(file)})
    examples, pairs, excluded = [], [], []
    for eid in example_ids:
        if eid not in probes:
            raise ValueError("requested discovery id missing from probes")
        e, eos = probes[eid]
        prompt, continuation = e['input_ids'], e['continuation_ids']
        if (not isinstance(prompt, list) or len(prompt) < 2 or not isinstance(continuation, list)
                or any(type(t) is not int or t < 0 for t in prompt + continuation)):
            raise ValueError("invalid explicit probe tokens")
        ref = refs[eid]['tokens']
        if ref[:len(prompt)] != prompt:
            raise ValueError("reference/probe prompt mismatch")
        content = []
        for t in continuation:
            if t in eos:
                break
            content.append(t)
        tokens = prompt + content
        if len(tokens) > 256 or len(ref) > 256:
            raise ValueError("context<=256 tokens required")
        counts, positions, witnesses = Counter(), [], []
        for end in range(ngram_size - 1, len(content)):
            gram = tuple(content[end-ngram_size+1:end+1])
            counts[gram] += 1
            target = len(prompt) + end
            if counts[gram] < min_occurrences or target >= len(ref):
                continue
            rg = tuple(ref[target-ngram_size+1:target+1])
            rc = sum(tuple(ref[j-ngram_size+1:j+1]) == rg for j in range(ngram_size-1, target+1))
            if rc > 1:
                continue
            positions.append(target-1)
            witnesses.append({'prediction_position': target-1, 'generated_ngram': list(gram),
                              'generated_occurrences': counts[gram], 'reference_occurrences': rc})
            if len(positions) == max_positions:
                break
        if not positions:
            excluded.append({'id': eid, 'reason': 'no eligible recurring ngram with same-position nonrecurring reference'})
            continue
        rid, fid = eid + '/reference', eid + '/generated'
        examples.extend([{'id': rid, 'tokens': ref}, {'id': fid, 'tokens': tokens}])
        pairs.append({'id': eid, 'reference_id': rid, 'failure_id': fid,
                      'reference_positions': positions, 'failure_positions': positions, 'witnesses': witnesses})
    if not pairs:
        raise ValueError("no actual recurring discovery contexts; no dataset written")
    result = {'name': 'observed-repetition-contexts', 'version': '1', 'splits': {'discovery': examples},
              'pairs': pairs, 'excluded': excluded, 'source_sha256': source_hash,
              'reference': {'path': reference_path, 'sha256': ds['sha256']}, 'probes': provenance,
              'ngram_size': ngram_size, 'min_occurrences': min_occurrences,
              'label_scope': 'Observed token recurrence only; same prompt and absolute position do not establish semantic equivalence or causation.',
              'model_forward_calls': 0, 'parameter_updates': 0}
    ev.write_report(root, result, output)
    return {'dataset_path': output, 'dataset_sha256': core.sha256(core.local(root, output)),
            'pairs': pairs, 'excluded': excluded, 'model_forward_calls': 0, 'parameter_updates': 0}
