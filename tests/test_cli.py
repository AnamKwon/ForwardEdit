import json
import os
import subprocess
import sys
from pathlib import Path
from test_diagnostics import tiny_world
from test_experiment import config
from weight_space_lab.core import write_candidate


def test_cli_real_commands(tmp_path):
    tiny_world(tmp_path)
    p = config(tmp_path, 0)
    (tmp_path / 'config.json').write_text(json.dumps(p))
    write_candidate(tmp_path, p['plans'][0], 'candidate')
    (tmp_path / 'options.json').write_text(json.dumps({'min_nll_gain': .001, 'bootstrap': 10}))
    env = {**os.environ, 'PYTHONPATH': str(Path.cwd()), 'PYTHONDONTWRITEBYTECODE': '1'}
    def run(*args):
        result = subprocess.run([sys.executable, '-B', '-m', 'weight_space_lab.cli',
            '--workspace', str(tmp_path), *args], env=env, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert run('baseline', 'model', 'data.json', '--splits', 'validation')['metrics']['validation']['tokens'] == 3
    assert run('diagnose', 'model', 'data.json', 'transformer.h.0.mlp.c_proj.weight')['tensors']
    assert run('evaluate', 'model', 'candidate', 'data.json', '--options', 'options.json')['behavior_validation'] == 'gate_failed'
    assert run('run-experiment', 'config.json', 'run')['status'] == 'no_winner'


def test_cli_import_is_lazy():
    result = subprocess.run([sys.executable, '-B', '-c',
        'import sys; import weight_space_lab.cli; import weight_space_lab.diagnostics; '
        'import weight_space_lab.experiment; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
