import torch


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
