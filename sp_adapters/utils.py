import torch
from torch import nn


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
        target.bias = nn.Parameter(maybe_clone(source.bias), requires_grad=False)
