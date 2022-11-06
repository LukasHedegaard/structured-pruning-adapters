import torch
from torch import nn

from sp_adapters import SPLoRA
from sp_adapters.splora import _DEFAULT_INIT_RANGE, SPLoRALinear

EPS = 1e-9


def test_splora_linear():
    rank = 2
    in_features = 9
    out_features = 4
    x = torch.randn(2, 9)
    y = torch.randn(2, 4)

    lin = nn.Linear(in_features, out_features)
    splin = SPLoRA(lin, rank)
    assert torch.equal(lin.weight, splin.weight)
    assert torch.equal(lin.bias, splin.bias)

    # Test __repr__
    assert (
        splin.__repr__()
        == "SPLoRALinear(\n  in_features=9, out_features=4, bias=True\n  (adapter): LowRankMatrix(num_filters=1, in_features=9, out_features=4, rank=2)\n)"
    )

    # Prepare optim
    optimizer = torch.optim.Adam(splin.parameters())
    optimizer.zero_grad()

    # Save for later comparison
    prev_adapter_cols = torch.clone(splin.adapter.cols)
    prev_adapter_rows = torch.clone(splin.adapter.rows)
    prev_bias = torch.clone(splin.adapter_bias)

    # Forwards are approx equal since adapters were initialized with near-zero values
    lin_out = lin.forward(x)
    splin.weight.requires_grad = True
    splin_out = splin.forward(x)
    assert not splin.weight.requires_grad  # adapter enforced requires_grad=False
    assert torch.allclose(lin_out, splin_out, atol=1e-3)

    # Train a bit
    mse = nn.MSELoss()
    loss = mse(splin_out, y)
    loss.backward()
    optimizer.step()

    # Linear params remain unchanged
    assert torch.equal(lin.weight, splin.weight)
    assert torch.equal(lin.bias, splin.bias)

    # Adapter params changed!
    assert not torch.equal(splin.adapter.cols, prev_adapter_cols)
    assert not torch.equal(splin.adapter.rows, prev_adapter_rows)
    assert not torch.equal(splin.adapter_bias, prev_bias)

    # Count params
    tot_params = sum([torch.Tensor([p.shape]).prod() for p in splin.parameters()])

    # Count masked params
    in_features_mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
    out_features_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    splin.configure_parameter_read(
        in_features_mask=in_features_mask, out_features_mask=out_features_mask
    )
    tot_masked_params = sum(
        [torch.Tensor([p.shape]).prod() for p in splin.parameters()]
    )

    assert tot_masked_params == tot_params - 5 * rank - 2 * rank

    # Export to lin
    lin2 = splin.to_module()
    assert torch.equal(lin2.weight, splin.adapted_weight)
    assert torch.equal(lin2.bias, splin.adapted_bias)


def test_splora_conv():
    batch_size = 2
    rank = 2
    in_channels = 9
    out_channels = 4
    kernel_size = 3
    A = 9

    for ConvNd in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        D = int(ConvNd.__name__[-2])
        x_shape = [batch_size, in_channels] + [A for _ in range(D)]
        y_shape = [batch_size, out_channels] + [A - 2 for _ in range(D)]
        x = torch.randn(x_shape)
        y = torch.randn(y_shape)
        bias = D == 2

        conv = ConvNd(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        spconv = SPLoRA(conv, rank)
        assert torch.equal(conv.weight, spconv.weight)
        if bias:
            assert torch.equal(conv.bias, spconv.bias)

        # Prepare optim
        optimizer = torch.optim.Adam(spconv.parameters())
        optimizer.zero_grad()

        # Save for later comparison
        prev_adapter_cols = torch.clone(spconv.adapter.cols)
        prev_adapter_rows = torch.clone(spconv.adapter.rows)
        if bias:
            prev_bias = torch.clone(spconv.adapter_bias)

        # Forwards are approx equal since adapters were initialized with near-zero values
        conv_out = conv.forward(x)
        spconv_out = spconv.forward(x)
        assert torch.allclose(conv_out, spconv_out, atol=1e-3)

        # Train a bit
        mse = nn.MSELoss()
        loss = mse(spconv_out, y)
        loss.backward()
        optimizer.step()

        # Linear params remain unchanged
        assert torch.equal(conv.weight, spconv.weight)
        if bias:
            assert torch.equal(conv.bias, spconv.bias)

        # Adapter params changed!
        assert not torch.equal(spconv.adapter.cols, prev_adapter_cols)
        assert not torch.equal(spconv.adapter.rows, prev_adapter_rows)
        if bias:
            assert not torch.equal(spconv.adapter_bias, prev_bias)

        # Count params
        tot_params = sum([torch.Tensor([p.shape]).prod() for p in spconv.parameters()])

        # Count masked params
        in_features_mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
        out_features_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

        spconv.configure_parameter_read(
            in_features_mask=in_features_mask, out_features_mask=out_features_mask
        )
        tot_masked_params = sum(
            [torch.Tensor([p.shape]).prod() for p in spconv.parameters()]
        )

        assert tot_masked_params == tot_params - 5 * rank - 2 * rank

        # Export to conv
        conv2 = spconv.to_module()
        assert torch.equal(conv2.weight, spconv.adapted_weight)
        if bias:
            assert torch.equal(conv2.bias, spconv.adapted_bias)


def test_splora_named_parameters():
    in_features = 9
    out_features = 4
    rank = 2

    splin1 = SPLoRALinear(in_features, out_features, rank=rank, bias=True)
    nps1 = {v[0] for v in splin1.named_parameters()}
    assert "adapter.rows" in nps1
    assert "adapter.cols" in nps1
    assert "adapter_bias" in nps1

    # Linear params exis, but do not show in named_parameters
    assert "weight" not in nps1
    assert "bias" not in nps1
    assert isinstance(splin1.weight, nn.Parameter)
    assert isinstance(splin1.bias, nn.Parameter)

    splin2 = SPLoRALinear(in_features, out_features, rank=rank, bias=False)
    nps2 = {v[0] for v in splin2.named_parameters()}
    assert "adapter.rows" in nps2
    assert "adapter.cols" in nps2
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
    anet = SPLoRA(net, rank=1 / 8)

    # Fractional rank is based on output_channels
    assert anet.seq[0].adapter.rank == round(512 / 8)
    assert anet.fc2.adapter.rank == round(10 / 8)

    assert isinstance(net.seq[0], torch.nn.Linear)
    assert isinstance(anet.seq[0], SPLoRALinear)  # Different module
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

    # Conversion only handles
