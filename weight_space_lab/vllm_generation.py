"""Optional optimized generation subprocess, launched only inside an MCP research job.

Teacher-forced observations continue to use the explicit Transformers worker.
No edit selection, fitting, training or model execution outside a forward job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from . import core, evaluation as ev
from . import research as api


def seed_for(identifier, sample, seed):
    raw = json.dumps([identifier, sample, seed], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**31)


def materialize(root, c, spec, which):
    """CPU-only exact merged checkpoint; immutable source and explicit patch remain authoritative."""
    original = ev._model_dir(root, c['model_dir'])
    candidate = spec['candidate_dir'] if which=='edited' else None
    if candidate:
        manifest, patch = ev.verify_candidate(root, c['model_dir'], candidate)
        digest = core.sha256(core.local(root, str(Path(candidate)/'replacements.safetensors')))
    else:
        manifest,patch={'report':{'changed_parameters':0}},{}
        digest='original-'+c['source_sha256']
    metadata_hashes={p.name:core.sha256(p) for p in original.iterdir() if p.is_file() and p.suffix in ('.json','.jinja')}
    metadata_digest=hashlib.sha256(json.dumps(metadata_hashes,sort_keys=True).encode()).hexdigest()
    target = core.local(root, 'materialized_models/'+digest+'-'+metadata_digest[:16])
    if target.exists():
        if (target/'model.safetensors').stat().st_mode & 0o222:
            (target/'model.safetensors').chmod(0o444)
        info = api.read(root, target/'materialized.json')
        if info['source_sha256'] != c['source_sha256'] or info['patch_sha256'] != digest or core.sha256(target/'model.safetensors') != info['merged_sha256']:
            raise ValueError('materialized checkpoint binding mismatch')
        return target, candidate
    parent = target.parent
    parent.mkdir(exist_ok=True)
    used = sum(p.stat().st_size for p in parent.glob('*/model.safetensors'))
    needed = (original/'model.safetensors').stat().st_size
    if used+needed > c.get('materialized_model_bytes_total', 100*1024**3):
        raise ValueError('materialized checkpoint budget exhausted')
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    stage = core.local(root, 'materialized_models/stage-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        state = {}
        with safe_open(original/'model.safetensors', framework='pt', device='cpu') as handle:
            for name in handle.keys():
                runtime_name = 'model.language_model.'+name[len('model.'):] if name.startswith('model.') else name
                state[runtime_name] = torch.from_numpy(patch[name]) if name in patch else handle.get_tensor(name)
            save_file(state, stage/'model.safetensors')
        (stage/'model.safetensors').chmod(0o444)
        for p in original.iterdir():
            if p.is_file() and p.suffix in ('.json','.jinja') and p.name!='config.json':
                shutil.copy2(p, stage/p.name)
        shutil.copy2(original/'conditional_config.json',stage/'config.json')
        info = {'source_sha256': c['source_sha256'], 'patch_sha256': digest,
                'merged_sha256': core.sha256(stage/'model.safetensors'),
                'changed_parameters': manifest['report']['changed_parameters'],
                'runtime_config_sha256':core.sha256(stage/'config.json'),
                'method': 'Exact CPU source tensors and explicit replacements, with model.* -> model.language_model.* runtime key mapping and the original official conditional config. Text-only engine; no vision module, fitting or value changes.'}
        (stage/'materialized.json').write_text(json.dumps(info))
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target, candidate


def _traps():
    from .inference_guards import install_inference_guards
    install_inference_guards()


# This module is also importable for CPU tests without the optional engine.
try:
    from vllm.v1.worker.gpu_worker import Worker as _Worker
except ImportError:
    _Worker = object


class InferenceOnlyGPUWorker(_Worker):
    def init_device(self, *args, **kwargs):
        _traps()
        return super().init_device(*args, **kwargs)

    def verify_direct_weights(self, root_string, model_dir, candidate):
        return verify_loaded_model(self.model_runner.model, root_string, model_dir, candidate)


def verify_loaded_model(model, root_string, model_dir, candidate):
    """Read actual GPU weights; reject unsupported packed targets rather than ignoring edits."""
    import numpy as np
    import torch
    from safetensors import safe_open
    root = Path(root_string)
    model.eval().requires_grad_(False)
    if candidate:
        manifest, patch = ev.verify_candidate(root, model_dir, candidate)
    else:
        manifest,patch={'report':{'changed_parameters':0}},{}
    parameters = dict(model.named_parameters())
    changed, targets = 0, []
    with safe_open(core.local(root, str(Path(model_dir)/'model.safetensors')), framework='np') as source:
        checked = patch or {name:source.get_tensor(name) for name in ('model.layers.0.mlp.down_proj.weight','model.layers.23.mlp.down_proj.weight')}
        for name, array in checked.items():
            runtime_names=[n for n in ('language_model.'+name,name) if n in parameters]
            if len(runtime_names)!=1:
                raise ValueError('vLLM target packed/renamed; no verified direct mapping: '+name)
            runtime_name=runtime_names[0]
            param = parameters[runtime_name]
            expected = torch.from_numpy(np.array(array,copy=True)).to(device=param.device,dtype=param.dtype)
            if tuple(param.shape) != tuple(expected.shape) or not torch.equal(param,expected):
                raise ValueError('actual vLLM GPU tensor differs from candidate: '+name)
            original = torch.from_numpy(source.get_tensor(name)).to(device=param.device,dtype=param.dtype)
            changed += int((expected != original).sum())
            targets.append(name)
    if manifest['report']['changed_parameters'] and not changed:
        raise ValueError('nonidentity edit disappears under vLLM inference casting')
    return {'actual_changed':changed,'verified_targets':targets,'requires_grad':any(p.requires_grad for p in model.parameters())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--which',choices=['original','edited'],required=True)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=args.workspace.resolve(strict=True)
    snap=api.load_snapshot(root,args.job)
    c,spec=snap['config'],snap['spec']
    api.verify_hashes(root,snap['hashes'])
    examples=api.read(root,args.inputs)
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE)
    needed=(ev._model_dir(root,c['model_dir'])/'model.safetensors').stat().st_size+1024**2
    resource.setrlimit(resource.RLIMIT_FSIZE,(max(soft,needed),hard))
    # The engine's own shared-memory ring/cache files are runtime allocations,
    # not research artifacts. Keep the separately bounded runtime allowance until
    # it finishes; restore the small job-output limit before writing its result.
    directory,candidate=materialize(root,c,spec,args.which)
    _traps()
    from vllm import LLM, SamplingParams
    # This engine release ships the official Qwen text config while its
    # Transformers dependency predates registration of the text-only type.
    started=time.monotonic()
    engine=LLM(model=str(directory),dtype='bfloat16',trust_remote_code=False,language_model_only=True,
        skip_tokenizer_init=True,tensor_parallel_size=1,distributed_executor_backend='mp',
        max_model_len=c['max_input_tokens']+c['max_new_tokens'],
        gpu_memory_utilization=c.get('vllm_gpu_memory_utilization',.80),
        max_num_seqs=c.get('batch_size',32),max_num_batched_tokens=8192,
        enforce_eager=c.get('vllm_enforce_eager',False),enable_prefix_caching=False,
        mamba_ssm_cache_dtype='float32',worker_cls='weight_space_lab.vllm_generation.InferenceOnlyGPUWorker')
    engine_ready=time.monotonic()
    verification=engine.collective_rpc('verify_direct_weights',args=(str(root),c['model_dir'],candidate))
    verified_at=time.monotonic()
    if len(verification)!=1 or verification[0]['requires_grad']:
        raise ValueError('vLLM must contain one frozen verified model')
    s=c['sampling'];requests=[];parameters=[];rows=[]
    config=json.loads((directory/'config.json').read_text())
    eos=config.get('eos_token_id',config.get('text_config',{}).get('eos_token_id'))
    eos=eos if isinstance(eos,list) else ([] if eos is None else [eos])
    if not eos:
        raise ValueError('explicit source EOS ids required with skip_tokenizer_init')
    for example in examples:
        for sample in range(spec['n']):
            requests.append({'prompt_token_ids':example['tokens']})
            parameters.append(SamplingParams(temperature=s['temperature'],top_p=s['top_p'],top_k=s['top_k'],
                presence_penalty=s['presence_penalty'],max_tokens=spec['max_new_tokens'],
                stop_token_ids=eos,seed=seed_for(example['id'],sample,s['seed']),detokenize=False))
            rows.append({'id':example['id'],'sample':sample})
    output=engine.generate(requests,parameters,use_tqdm=False)
    generated_at=time.monotonic()
    if len(output)!=len(rows):
        raise ValueError('vLLM output count mismatch')
    for row,request in zip(rows,output):
        if len(request.outputs)!=1:
            raise ValueError('one actual sample per item required')
        result=request.outputs[0]
        if result.finish_reason not in ('stop','length'):
            raise ValueError('engine did not finish normally')
        row.update(continuation_ids=list(result.token_ids),finish_reason='eos' if result.finish_reason=='stop' else 'max_new_tokens')
    api.verify_hashes(root,snap['hashes'])
    payload={'rows':rows,'verification':verification[0],'wall_seconds':time.monotonic()-started,
             'backend':'vllm-'+__import__('importlib.metadata',fromlist=['version']).version('vllm'),'engine_generation_calls':1,'eos_token_ids':eos,
             'engine_start_seconds':engine_ready-started,'gpu_verification_seconds':verified_at-engine_ready,
             'generation_seconds':generated_at-verified_at,
             'model_forward_counter_scope':'Engine-internal autoregressive/profiling invocations are not observable here; logical pass and token/sample counts are reported separately.'}
    resource.setrlimit(resource.RLIMIT_FSIZE,(soft,hard))
    api.atomic(root,args.output,payload)


if __name__=='__main__':
    main()
