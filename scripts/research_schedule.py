"""Run a fixed, host-authored research-forward schedule through MCP.

Job IDs are persisted before waiting. Resuming never silently resubmits a step;
an ambiguous submission requires reconciliation of the existing server job.
This client selects no parameters, seeds, candidates or evaluation thresholds.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


async def run_step(call, workspace, receipts, step, *, poll_seconds=10,
                   timeout_seconds=10800, emit=lambda value: None,
                   clock=time.monotonic, sleep=asyncio.sleep):
    workspace, receipts = Path(workspace).resolve(), Path(receipts).resolve()
    name, spec = step['name'], step['spec']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('step name must be a simple file-safe identifier')
    receipts.mkdir(parents=True, exist_ok=True)
    path = receipts / (name + '.json')
    binding = {'workspace': str(workspace), 'spec_sha256': digest(spec),
               'config_sha256': hashlib.sha256((workspace / 'research.json').read_bytes()).hexdigest()}
    if path.exists():
        record = json.loads(path.read_text())
        if any(record.get(k) != v for k, v in binding.items()):
            raise ValueError('existing step binding differs; preserve its job, do not resubmit')
        if not record.get('job_id'):
            raise RuntimeError('uncertain or rejected submission; reconcile the retained intent before continuing: ' + str(path))
    else:
        # An interruption after server acceptance but before the ID is received
        # must not turn a restart into an additional model call.
        record = {**binding, 'name': name, 'spec': spec, 'status': 'submitting',
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save(path, record)
        try:
            response = await call('research_forward', {'spec': spec})
        except Exception as exc:
            record['submission_error'] = str(exc)[:500]
            save(path, record)
            raise
        record['submission'] = response
        result = response.get('result', {})
        if response.get('isError') or not isinstance(result, dict) or not re.fullmatch(r'[0-9a-f]{32}', result.get('job_id', '')):
            record['status'] = 'rejected'
            save(path, record)
            raise RuntimeError('MCP rejected the request; retained at ' + str(path))
        record.update(job_id=result['job_id'], status=result.get('status', 'pending'))
        save(path, record)
        emit({'name': name, 'job_id': record['job_id'], 'status': 'submitted'})

    job_id = record['job_id']
    deadline = clock() + timeout_seconds
    while True:
        response = await call('research_status', {'job_id': job_id})
        if response.get('isError'):
            raise RuntimeError('status failed; retain existing job ' + job_id)
        state = response['result']['status']
        record['status'] = state
        save(path, record)
        emit({'name': name, 'job_id': job_id, 'status': state})
        if state in ('succeeded', 'failed'):
            verified = await call('research_result', {'job_id': job_id})
            record['result'] = verified
            if (verified.get('isError') or verified.get('result', {}).get('job_id') != job_id
                    or verified.get('result', {}).get('status') == 'failed' or state == 'failed'):
                record['status'] = 'failed'
                save(path, record)
                raise RuntimeError('research job failed or could not be verified: ' + job_id)
            record['status'] = 'succeeded'
            save(path, record)
            return verified['result']
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError('job remains live; resume this receipt instead of resubmitting: ' + job_id)
        await sleep(min(poll_seconds, remaining))


async def run_schedule(call, workspace, receipts, schedule, **options):
    steps = schedule['steps']
    if not isinstance(steps, list) or not steps or len({s['name'] for s in steps}) != len(steps):
        raise ValueError('a nonempty fixed schedule with unique step names is required')
    results = []
    for step in steps:
        results.append(await run_step(call, workspace, receipts, step, **options))
    return results


async def cli(args):
    # Import optional MCP dependencies only for live calls, never for CPU tests.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import fcntl

    workspace, receipts = args.workspace.resolve(), args.receipts.resolve()
    config = json.loads((workspace / 'research.json').read_text())
    if config.get('device') == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('set CUDA_VISIBLE_DEVICES to the designated inference GPU')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from weight_space_lab.research import config as checked_config, validate_spec
    schedule = json.loads(args.schedule.read_text())
    checked = checked_config(workspace)
    # Reuse the server's read-only validator instead of inventing a different
    # client schema. In particular, paired teacher passes are separate jobs.
    for step in schedule['steps']:
        validate_spec(workspace, checked, step['spec'])
    receipts.mkdir(parents=True, exist_ok=True)
    lock = (receipts / '.schedule.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('another client owns this receipt directory')
    wrapper = '''import torch,runpy
def forbidden(*a,**k):
    raise RuntimeError('Training/autograd/optimizer is forbidden')
torch.Tensor.backward=forbidden
torch.autograd.backward=forbidden
torch.autograd.grad=forbidden
torch.optim.Optimizer.__init__=forbidden
torch.set_grad_enabled(False)
runpy.run_module('weight_space_lab.server',run_name='__main__')
'''
    env = {**os.environ, 'PYTHONPATH': str(project), 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'USE_TF': '0'}
    params = StdioServerParameters(command=sys.executable,
        args=['-B', '-c', wrapper, '--workspace', str(workspace)], env=env)
    try:
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(tool, arguments):
                    response = await session.call_tool(tool, arguments)
                    blocks = [c.text for c in response.content if getattr(c, 'type', None) == 'text']
                    try:
                        value = json.loads(blocks[0])
                    except (IndexError, json.JSONDecodeError):
                        value = {'error': '\n'.join(blocks)}
                    return {'tool': tool, 'arguments': arguments,
                            'isError': bool(response.isError), 'result': value}

                results = await run_schedule(call, workspace, receipts,
                    schedule, poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds,
                    emit=lambda row: print(json.dumps(row), flush=True))
                print(json.dumps({'completed_steps': len(results), 'receipts': str(receipts)}))
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--timeout-seconds', type=float, default=10800)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error('poll/timeout must be positive')
    try:
        asyncio.run(cli(args))
    except KeyboardInterrupt:
        parser.exit(130, 'Client interrupted; retain receipts and resume the existing jobs.\n')
    except Exception as exc:
        def messages(error):
            children = getattr(error, 'exceptions', None)
            return [m for child in children for m in messages(child)] if children else [str(error)]
        parser.exit(1, '\n'.join(messages(exc)) + '\n')


if __name__ == '__main__':
    main()
