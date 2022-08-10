import torch
from torch import nn

from sp_adapters import SPLoPA
from sp_adapters.splopa import SPLoPALinear


def test_splopa():
    num_prototypes = 2
    n, m = 4, 9
    p, q = 2, 3
    x = torch.randn(2, 9)
    y = torch.randn(2, 4)

    lin = nn.Linear(m, n)
    splin = SPLoPA(lin, num_prototypes=num_prototypes, block_shape=(p, q))
    assert torch.equal(lin.weight, splin.weight)
    assert torch.equal(lin.bias, splin.bias)

    # Prepare optim
    optimizer = torch.optim.Adam(splin.parameters())
    optimizer.zero_grad()

    # Save for later comparison
    prev_prototype_cols = torch.clone(splin.adapter.prototypes.cols)
    prev_prototype_rows = torch.clone(splin.adapter.prototypes.rows)
    prev_pos_weights = torch.clone(splin.adapter.pos_weights)
    prev_bias = torch.clone(splin.adapter_bias)

    # Forwards are approx equal since adapters were initialized with near-zero values
    lin_out = lin.forward(x)
    splin_out = splin.forward(x)
    assert torch.allclose(lin_out, splin_out, atol=1e-5)

    # Train a bit
    mse = nn.MSELoss()
    loss = mse(splin_out, y)
    loss.backward()
    optimizer.step()

    # Linear params remain unchanged
    assert torch.equal(lin.weight, splin.weight)
    assert torch.equal(lin.bias, splin.bias)

    # Adapter params changed!
    assert not torch.equal(splin.adapter.prototypes.cols, prev_prototype_cols)
    assert not torch.equal(splin.adapter.prototypes.rows, prev_prototype_rows)
    assert not torch.equal(splin.adapter.pos_weights, prev_pos_weights)
    assert not torch.equal(splin.adapter_bias, prev_bias)

    # A new SPLoPA Linear will share prototypes with prior instances but not pos_weights
    splin2 = SPLoPA(lin, num_prototypes=num_prototypes, block_shape=(p, q))
    assert torch.equal(splin.adapter.prototypes.rows, splin2.adapter.prototypes.rows)
    assert torch.equal(splin.adapter.prototypes.cols, splin2.adapter.prototypes.cols)
    assert not torch.equal(splin.adapter.pos_weights, splin2.adapter.pos_weights)
    assert not torch.equal(splin.adapter_bias, splin2.adapter_bias)

    # Count params
    tot_params = sum([torch.Tensor([p.shape]).prod() for p in splin.parameters()])

    # Count masked params
    block_mask = torch.tensor(  # 50% masking
        [[[1, 0, 1], [0, 0, 1]], [[0, 1, 0], [1, 1, 0]]],
        dtype=torch.bool,
    )
    splin.configure_parameter_read(mask=block_mask)
    tot_masked_params = sum(
        [torch.Tensor([p.shape]).prod() for p in splin.parameters()]
    )

    assert tot_masked_params == tot_params - num_prototypes * (n // p * m // q) * 0.5


def test_splopa_named_parameters():
    num_prototypes = 2
    n, m = 4, 9
    p, q = 2, 3

    splin1 = SPLoPALinear(
        m, n, num_prototypes=num_prototypes, block_shape=(p, q), bias=True
    )
    nps1 = {v[0] for v in splin1.named_parameters()}
    assert "adapter.prototypes.rows" in nps1
    assert "adapter.prototypes.cols" in nps1
    assert "adapter.pos_weights" in nps1
    assert "adapter_bias" in nps1

    # Linear params exis, but do not show in named_parameters
    assert "weight" not in nps1
    assert "bias" not in nps1
    assert isinstance(splin1.weight, nn.Parameter)
    assert isinstance(splin1.bias, nn.Parameter)

    splin2 = SPLoPALinear(
        m, n, num_prototypes=num_prototypes, block_shape=(p, q), bias=False
    )
    nps2 = {v[0] for v in splin2.named_parameters()}
    assert "adapter.prototypes.rows" in nps2
    assert "adapter.prototypes.cols" in nps2
    assert "adapter.pos_weights" in nps2
    assert "adapter_bias" not in nps2  # <==
    assert splin2.adapter_bias is None

    # Linear params exis, but do not show in named_parameters
    assert "weight" not in nps2
    assert "bias" not in nps2
    assert isinstance(splin2.weight, nn.Parameter)
    assert splin2.bias is None
