"""Real stdio round trip against the optional SDK."""
import asyncio
import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def is_error(result):
    # SDK2 uses snake_case Python fields; the MCP wire contract is unchanged.
    return result.is_error if hasattr(result, 'is_error') else result.isError


def test_stdio_candidate_workflow(tmp_path):
    save_file({"linear.weight": np.eye(2, dtype=np.float32)}, str(tmp_path / "model.safetensors"))

    async def run():
        params = StdioServerParameters(command=sys.executable, args=[
            "-m", "weight_space_lab.server", "--workspace", str(tmp_path)])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert {tool.name for tool in listing.tools} == {
                    "inspect_checkpoint", "read_tensor_slice", "preview_edit", "create_candidate",
                    "evaluate_baseline", "diagnose_model", "evaluate_candidate", "run_experiment",
                    "describe_parameter", "write_parameter_values", "score_candidate", "probe_model",
                    "analyze_parameter_contexts", "prepare_edit_proposal", "confirm_generation",
                    "compare_candidate_parameters", "read_parameter_entries", "describe_model_structure",
                    "inspect_module_computation", "analyze_matrix_geometry", "trace_token_flow", "revise_candidate",
                    "build_repetition_contexts", "research_forward", "research_status", "research_result",
                    "research_read_array", "research_read_generations", "research_prepare_edit", "research_make_contexts",
                    "research_export_tensor", "research_compare_generations"}
                result = await session.call_tool("inspect_checkpoint", {"checkpoint": "model.safetensors"})
                assert not is_error(result)
                info = json.loads(result.content[0].text)
                plan = {"checkpoint": "model.safetensors", "source_sha256": info["sha256"],
                        "hypothesis": "stdio round trip", "limits": {
                            "max_changed_fraction": .5, "max_relative_l2": .5},
                        "operations": [{"tensor": "linear.weight", "kind": "scale", "axis": 0,
                                        "indices": [0], "strength": .25}]}
                preview = await session.call_tool("preview_edit", {"plan": plan})
                assert not is_error(preview)
                assert json.loads(preview.content[0].text)["changed_parameters"] == 1
                saved = await session.call_tool("create_candidate", {"plan": plan, "output": "candidate"})
                assert not is_error(saved)
                assert (tmp_path / "candidate/manifest.json").is_file()
                rejected = await session.call_tool("inspect_checkpoint", {"checkpoint": "../outside.safetensors"})
                assert is_error(rejected)
    asyncio.run(asyncio.wait_for(run(), timeout=45))


def test_stdio_lm_workflow(tmp_path):
    from test_diagnostics import tiny_world
    from test_experiment import config
    from weight_space_lab.core import write_candidate
    tiny_world(tmp_path)
    c = config(tmp_path, 0)
    write_candidate(tmp_path, c['plans'][0], 'candidate')

    async def run():
        # Tripwires also live in the real server subprocess, not just this test.
        wrapper = ('import torch,runpy; '
            'exec("def forbidden(*a, **k):\\n    raise AssertionError(\'training forbidden\')"); '
            'torch.Tensor.backward=forbidden; torch.autograd.backward=forbidden; '
            'torch.autograd.grad=forbidden; torch.optim.Optimizer.__init__=forbidden; '
            'runpy.run_module("weight_space_lab.server",run_name="__main__")')
        params = StdioServerParameters(command=sys.executable, args=[
            '-B', '-c', wrapper, '--workspace', str(tmp_path)])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                shared = {'model_dir': 'model', 'dataset_path': 'data.json'}
                base = await session.call_tool('evaluate_baseline', {**shared, 'splits': ['validation']})
                assert not is_error(base)
                assert json.loads(base.content[0].text)['metrics']['validation']['tokens'] == 3
                diag = await session.call_tool('diagnose_model', {**shared,
                    'tensors': ['transformer.h.0.mlp.c_proj.weight']})
                assert not is_error(diag)
                assert json.loads(diag.content[0].text)['mode'] == 'direct_inference_diagnostic'
                described = await session.call_tool('describe_parameter', {'model_dir': 'model',
                    'tensor': 'transformer.h.0.mlp.c_proj.weight', 'rows': [0], 'columns': [0, 1]})
                assert not is_error(described)
                d = json.loads(described.content[0].text)
                vector = await session.call_tool('read_parameter_entries', {
                    'checkpoint': 'model/model.safetensors', 'tensor': 'transformer.h.0.ln_1.weight',
                    'flat_indices': [0, 1]})
                assert not is_error(vector)
                assert len(json.loads(vector.content[0].text)['entries']) == 2
                structure = await session.call_tool('describe_model_structure', {
                    'model_dir': 'model', 'prefix': 'transformer.h.0', 'max_modules': 2})
                assert not is_error(structure) and json.loads(structure.content[0].text)['next_offset'] == 2
                source = await session.call_tool('inspect_module_computation', {
                    'model_dir': 'model', 'modules': ['transformer.h.0.mlp', d['owner']]})
                assert not is_error(source)
                assert json.loads(source.content[0].text)['modules'][1]['equation'] == 'y = x @ W + bias; W[in,out]'
                geometry = await session.call_tool('analyze_matrix_geometry', {
                    'checkpoint': 'model/model.safetensors', 'tensor': d['tensor'],
                    'rows': [0, 1], 'columns': [0, 1], 'rank': 2})
                assert not is_error(geometry)
                traced_flow = await session.call_tool('trace_token_flow', {**shared,
                    'selections': [{'module': d['owner'], 'input_features': [0, 1], 'output_features': [0, 1]}],
                    'examples': [{'id': 'd', 'positions': [0, 1]}], 'output': 'flow.json'})
                assert not is_error(traced_flow)
                assert not json.loads(traced_flow.content[0].text)['report']['gradients_enabled']
                new_values = [[x + .001 for x in d['values'][0]]]
                written = await session.call_tool('write_parameter_values', {
                    'checkpoint': 'model/model.safetensors', 'source_sha256': d['source_sha256'],
                    'tensor': d['tensor'], 'rows': [0], 'columns': [0, 1],
                    'expected_values': d['values'], 'values': new_values,
                    'hypothesis': 'LLM-specified numeric edit, no gradient/optimizer',
                    'limits': {'max_changed_fraction': .1, 'max_relative_l2': .1}, 'output': 'direct'})
                assert not is_error(written)
                assert json.loads(written.content[0].text)['changed_parameters'] == 2
                after = await session.call_tool('describe_parameter', {'model_dir': 'model',
                    'tensor': d['tensor'], 'rows': [0], 'columns': [0, 1], 'candidate_dir': 'direct'})
                assert not is_error(after)
                np.testing.assert_allclose(json.loads(after.content[0].text)['values'], new_values, atol=1e-8)
                flow = await session.call_tool('compare_candidate_parameters', {
                    'model_dir': 'model', 'candidate_dir': 'direct'})
                assert not is_error(flow)
                traced = json.loads(flow.content[0].text)
                assert traced['new_changes'] == traced['net_changed'] == 2
                assert traced['model_forward_calls'] == 0
                same = await session.call_tool('compare_candidate_parameters', {
                    'model_dir': 'model', 'candidate_dir': 'direct', 'parent_candidate_dir': 'direct'})
                assert not is_error(same) and json.loads(same.content[0].text)['incremental_changed'] == 0
                current = json.loads(after.content[0].text)['values'][0][0]
                revision = await session.call_tool('revise_candidate', {
                    'model_dir': 'model', 'parent_candidate_dir': 'direct',
                    'operations': [{'tensor': d['tensor'], 'kind': 'set_values', 'rows': [0], 'columns': [0],
                        'expected_values': [[current]], 'values': [[current + .0005]]}],
                    'hypothesis': 'explicit subsequent fixture correction',
                    'rationale': {k: 'fixture arithmetic only' for k in (
                        'observation', 'hypothesis', 'value_derivation', 'prediction', 'disconfirming_result')},
                    'limits': {'max_changed_fraction': .1, 'max_relative_l2': .1}, 'output': 'revision'})
                assert not is_error(revision)
                traced = await session.call_tool('compare_candidate_parameters', {
                    'model_dir': 'model', 'candidate_dir': 'revision', 'parent_candidate_dir': 'direct'})
                assert not is_error(traced) and json.loads(traced.content[0].text)['retained_parent_changes'] == 1
                probe = await session.call_tool('probe_model', {**shared, 'candidate_dir': 'direct',
                    'max_new_tokens': 2})
                assert not is_error(probe)
                assert json.loads(probe.content[0].text)['optimizer_steps'] == 0
                score = await session.call_tool('score_candidate', {**shared, 'candidate_dir': 'direct'})
                assert not is_error(score)
                assert set(json.loads(score.content[0].text)['metrics']) == {'discovery', 'validation'}
                analysis = await session.call_tool('analyze_parameter_contexts', {**shared,
                    'tensor': d['tensor'], 'rows': [0], 'columns': [0, 1], 'proposed_values': new_values,
                    'pairs': [{'id': 'same-context-control', 'reference_id': 'd', 'failure_id': 'd',
                               'reference_positions': [0], 'failure_positions': [0]}], 'output': 'analysis.json'})
                assert not is_error(analysis)
                a = json.loads(analysis.content[0].text)
                assert a['report']['evidence_status'] == 'observational_only'
                assert a['report']['observations'][0]['identical_prefix']
                proposal = await session.call_tool('prepare_edit_proposal', {
                    'analysis_path': 'analysis.json', 'basis': {},
                    'limits': {'max_changed_fraction': .1, 'max_relative_l2': .1}, 'output': 'proposal.json'})
                assert not is_error(proposal)
                proposed = json.loads(proposal.content[0].text)
                assert proposed['decision'] == 'defer' and proposed['plan'] is None
                assert not proposed['candidate_created']
                result = await session.call_tool('evaluate_candidate', {**shared,
                    'candidate_dir': 'candidate', 'options': {'min_nll_gain': .001, 'bootstrap': 10}})
                assert not is_error(result)
                assert json.loads(result.content[0].text)['behavior_validation'] == 'gate_failed'
                experiment = await session.call_tool('run_experiment', {'config': c, 'output': 'run'})
                assert not is_error(experiment)
                assert json.loads(experiment.content[0].text)['status'] == 'no_winner'
    asyncio.run(asyncio.wait_for(run(), timeout=45))
