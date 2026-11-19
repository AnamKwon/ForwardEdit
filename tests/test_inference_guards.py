"""A compiler lookup may run after guards; training must still be blocked."""
import subprocess
import sys


def test_guarded_compiler_lookup_still_blocks_training():
    code = '''
import torch
from weight_space_lab.inference_guards import install_inference_guards
install_inference_guards()
from torch._dynamo.trace_rules import get_torch_obj_rule_map
if hasattr(get_torch_obj_rule_map, 'cache_clear'):
    get_torch_obj_rule_map.cache_clear()
get_torch_obj_rule_map()
for fn in (torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad, torch.optim.Optimizer.__init__):
    try:
        fn(None)
    except RuntimeError as exc:
        assert 'forbidden' in str(exc)
    else:
        raise AssertionError('a training API was not blocked')
assert not torch.is_grad_enabled()
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
