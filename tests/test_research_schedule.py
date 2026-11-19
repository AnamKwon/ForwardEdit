"""Client lifecycle controls: no real model, MCP process or parameter updates."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('research_schedule',
    Path(__file__).resolve().parents[1] / 'scripts/research_schedule.py')
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


class RPC:
    def __init__(self, state='pending', disconnect=False):
        self.state, self.disconnect, self.calls = state, disconnect, []

    async def __call__(self, tool, arguments):
        self.calls.append(tool)
        if tool == 'research_forward' and self.disconnect:
            raise ConnectionError('reply lost after server accepted the job')
        result = {'job_id': 'a' * 32}
        if tool in ('research_forward', 'research_status'):
            result['status'] = 'pending' if tool == 'research_forward' else self.state
        elif self.state == 'failed':
            result['status'] = 'failed'
        return {'isError': False, 'result': result}


@pytest.fixture
def paths(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'research.json').write_text('{}')
    return workspace, tmp_path / 'receipts'


def step():
    return {'name': 'fixed_probe', 'spec': {'type': 'generate', 'model': 'original'}}


def test_timeout_persists_id_and_resume_does_not_forward(paths):
    workspace, receipts = paths
    rpc = RPC()
    with pytest.raises(TimeoutError, match='resume'):
        asyncio.run(runner.run_step(rpc, workspace, receipts, step(), timeout_seconds=0))
    assert json.loads((receipts / 'fixed_probe.json').read_text())['job_id'] == 'a' * 32
    rpc.state = 'succeeded'
    asyncio.run(runner.run_step(rpc, workspace, receipts, step()))
    assert rpc.calls.count('research_forward') == 1
    assert json.loads((receipts / 'fixed_probe.json').read_text())['status'] == 'succeeded'


def test_lost_submission_reply_is_not_silently_resubmitted(paths):
    workspace, receipts = paths
    rpc = RPC(disconnect=True)
    with pytest.raises(ConnectionError):
        asyncio.run(runner.run_step(rpc, workspace, receipts, step()))
    assert json.loads((receipts / 'fixed_probe.json').read_text())['status'] == 'submitting'
    with pytest.raises(RuntimeError, match='uncertain'):
        asyncio.run(runner.run_step(rpc, workspace, receipts, step()))
    assert rpc.calls == ['research_forward']


def test_changed_spec_cannot_reuse_or_resubmit_a_completed_step(paths):
    workspace, receipts = paths
    rpc = RPC('succeeded')
    asyncio.run(runner.run_step(rpc, workspace, receipts, step()))
    count = len(rpc.calls)
    changed = {**step(), 'spec': {'type': 'generate', 'model': 'edited'}}
    with pytest.raises(ValueError, match='binding differs'):
        asyncio.run(runner.run_step(rpc, workspace, receipts, changed))
    assert len(rpc.calls) == count


def test_failed_job_is_preserved_and_stops_remaining_schedule(paths):
    workspace, receipts = paths
    rpc = RPC('failed')
    schedule = {'steps': [step(), {**step(), 'name': 'must_not_start'}]}
    with pytest.raises(RuntimeError, match='job failed'):
        asyncio.run(runner.run_schedule(rpc, workspace, receipts, schedule))
    assert rpc.calls.count('research_forward') == 1
    assert json.loads((receipts / 'fixed_probe.json').read_text())['status'] == 'failed'
    assert not (receipts / 'must_not_start.json').exists()
