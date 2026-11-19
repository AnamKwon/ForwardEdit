"""All active tests obey the inference-only project contract."""
import pytest


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    try:
        import torch
    except ImportError:
        return

    def forbidden(*args, **kwargs):
        raise AssertionError("Training/autograd/optimizer is forbidden in weight-space experiments")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
