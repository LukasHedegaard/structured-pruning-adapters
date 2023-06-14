from copy import deepcopy
from logging import getLogger
from typing import Callable, Type

import torch
from torch import nn

logger = getLogger(__name__)


def bkron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    This operation corresponds to per-instance torch.kron along dim 0.
    Credits: https://github.com/yulkang/pylabyk
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def copy_linear_params_(source: nn.Linear, target: nn.Linear, clone=True):
    maybe_clone = torch.clone if clone else lambda x: x
    target.weight = nn.Parameter(maybe_clone(source.weight), requires_grad=False)
    if source.bias is not None:
        target.bias = nn.Parameter(maybe_clone(source.bias))


def recursive_replace(
    module: nn.Module,
    OldModuleType: Type[nn.Module],
    new_module_constructor: Callable[[nn.Module], nn.Module],
    inplace=False,
    _module_name=None,
    *args,
    **kwargs,
) -> nn.Module:
    assert isinstance(module, nn.Module), "Only a `torch.nn.Module` can be adapted"
    if not inplace:
        module = deepcopy(module)

    if isinstance(module, OldModuleType):
        return new_module_constructor(module, *args, **kwargs)

    # Recursively update children
    for n, c in module.named_children():
        full_name = ".".join(filter(None, [_module_name, n]))
        try:
            setattr(
                module,
                n,
                recursive_replace(
                    c,
                    OldModuleType,
                    new_module_constructor,
                    True,
                    full_name,
                    *args,
                    **kwargs,
                ),
            )
        except Exception as e:
            logger.warning(f"Unable to convert '{full_name}' ({c}): {e}")

    return module
