import torch
from torch import nn

from sp_adapters import SPLoPA
from sp_adapters.splopa import _DEFAULT_INIT_RANGE, SPLoPALinear

EPS = 1e-8


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
    splin.weight.requires_grad = True
    splin_out = splin.forward(x)
    assert not splin.weight.requires_grad  # adapter enforced requires_grad=False
    assert torch.allclose(lin_out, splin_out, atol=_DEFAULT_INIT_RANGE * 10)

    # The tollerance is determined by the initialisation
    splin2 = SPLoPA(
        lin,
        num_prototypes=num_prototypes,
        block_shape=(p, q),
        init_range=1e-6,
        shared_prototypes=False,
        shared_pos_weights=True,
    )
    splin2_out = splin2.forward(x)
    assert torch.allclose(lin_out, splin2_out, atol=1e-6 * 10)

    # Prototype weight sharing also works
    splin3 = SPLoPA(
        lin,
        num_prototypes=num_prototypes,
        block_shape=(p, q),
        init_range=1e-6,
        shared_prototypes=False,
        shared_pos_weights=True,
    )
    assert not torch.equal(
        splin3.adapter.prototypes.cols, splin2.adapter.prototypes.cols
    )
    assert not torch.equal(
        splin3.adapter.prototypes.rows, splin2.adapter.prototypes.rows
    )
    assert torch.equal(splin3.adapter.pos_weights, splin2.adapter.pos_weights)

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

    # Export to lin
    lin2 = splin.to_module()
    assert torch.equal(lin2.weight, splin.adapted_weight)
    assert torch.equal(lin2.bias, splin.adapted_bias)


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


def test_conversion():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, 512), torch.nn.ReLU()
            )
            self.fc2 = torch.nn.Linear(512, 10)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = self.seq(x)
            x = self.fc2(x)
            output = torch.nn.functional.log_softmax(x, dim=1)
            return output

    net = Net()

    # Convert
    anet = SPLoPA(net, block_shape=(16, 16))

    assert isinstance(net.seq[0], torch.nn.Linear)
    assert isinstance(anet.seq[0], SPLoPALinear)  # Different module
    assert isinstance(net.fc2, torch.nn.Linear)

    # Not converted since `out_featues=10` is incompatible with `block_shape=(16,16)`
    assert isinstance(anet.fc2, torch.nn.Linear)

    # Same weight
    assert torch.equal(net.seq[0].weight, anet.seq[0].weight)
    assert torch.equal(net.seq[0].bias, anet.seq[0].bias)

    # Adapted weight close but not equal
    assert torch.allclose(
        net.seq[0].weight, anet.seq[0].adapted_weight, atol=_DEFAULT_INIT_RANGE + EPS
    )
    assert not torch.equal(net.seq[0].weight, anet.seq[0].adapted_weight)

    assert torch.allclose(
        net.seq[0].bias, anet.seq[0].adapted_bias, atol=_DEFAULT_INIT_RANGE + EPS
    )
    assert not torch.equal(net.seq[0].bias, anet.seq[0].adapted_bias)
