import torch
from sp_adapters.utils import bkron


def test_kron():
    a = torch.arange(12).reshape(2, 2, 3)
    b = torch.arange(8).reshape(2, 2, 2) * 10

    res = bkron(a, b)

    assert torch.equal(res[0], torch.kron(a[0], b[0]))
    assert torch.equal(res[1], torch.kron(a[1], b[1]))
