"""Shims for running the transformers 5 Higgs/Xcodec code on transformers 4.57."""

import torch
from torch.nn import init as _torch_init


class _Init:
    """transformers 5 `initialization` module: torch.nn.init plus `copy_`."""

    def __getattr__(self, name):
        return getattr(_torch_init, name)

    @staticmethod
    def copy_(tensor, value):
        with torch.no_grad():
            tensor.copy_(value)
        return tensor


init = _Init()


def conv1d_output_length(module: torch.nn.Conv1d, input_length: int) -> int:
    return int(
        (input_length + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
        / module.stride[0]
        + 1
    )


def auto_docstring(obj=None, **_kwargs):
    """No-op stand-in: transformers 4.57's auto_docstring rejects `X | None` annotations."""
    if obj is None:
        return lambda o: o
    return obj
