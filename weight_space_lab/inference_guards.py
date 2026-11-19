"""Inference-only tripwires with distinct identities for Torch compiler registries."""

def install_inference_guards():
    import torch
    torch.set_grad_enabled(False)
    def guard(label):
        def forbidden(*args, **kwargs):
            raise RuntimeError('Training/autograd/optimizer is forbidden: ' + label)
        forbidden.__name__ = 'inference_only_' + label.replace('.', '_')
        return forbidden
    # Torch Dynamo assigns different rules to these public APIs. Aliasing all
    # of them to one callable causes a duplicate-object rule conflict on startup.
    for owner, name, label in (
        (torch.Tensor, 'backward', 'Tensor.backward'),
        (torch.autograd, 'backward', 'autograd.backward'),
        (torch.autograd, 'grad', 'autograd.grad'),
        (torch.optim.Optimizer, '__init__', 'Optimizer.__init__'),
    ):
        setattr(owner, name, guard(label))
