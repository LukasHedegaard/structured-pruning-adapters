import torch
from torch import nn

from sp_adapters import SPPaRA
from sp_adapters.sppara import (
    _DEFAULT_INIT_RANGE,
    SPPaRAConv1d,
    named_parameters,
    parameters,
)

EPS = 1e-9


def test_SPPaRA_conv():
    batch_size = 2
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
        spconv = SPPaRA(conv)
        assert torch.equal(conv.weight, spconv.weight)
        if bias:
            assert torch.equal(conv.bias, spconv.bias)

        # Prepare optim
        optimizer = torch.optim.Adam(spconv.parameters())
        optimizer.zero_grad()

        # Save for later comparison
        prev_adapter = torch.clone(spconv.adapter)

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

        # Adapter params changed!
        if bias:
            assert not torch.equal(conv.bias, spconv.bias)
        assert not torch.equal(spconv.adapter, prev_adapter)

        # Count params
        # tot_params = sum([torch.Tensor([p.shape]).prod() for p in spconv.parameters()])

        # Count masked params
        in_features_mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
        out_features_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

        tot_masked_params = sum(
            [
                torch.Tensor([p.shape]).prod()
                for p in parameters(
                    spconv,
                    adapter_weights_only=True,
                    in_features_mask=in_features_mask,
                    out_features_mask=out_features_mask,
                )
            ]
        )

        if bias:
            assert tot_masked_params == 4 * 2 + 2
        else:
            assert tot_masked_params == 4 * 2

        # Export to conv
        conv2 = spconv.to_module()
        assert torch.equal(conv2.weight, spconv.adapted_weight)
        if bias:
            assert torch.equal(conv2.bias, spconv.bias)


def test_SPPaRA_named_parameters():
    in_features = 9
    out_features = 4
    kernel_size = 3

    spconv1 = SPPaRAConv1d(in_features, out_features, kernel_size, bias=True)
    nps1 = {v[0] for v in named_parameters(spconv1)}
    assert "adapter" in nps1

    # Linear params exist, but do not show in named_parameters
    assert "weight" not in nps1
    assert isinstance(spconv1.weight, nn.Parameter)
    assert isinstance(spconv1.bias, nn.Parameter)


def test_conversion():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv1d(28 * 28, 512, 3), torch.nn.ReLU()
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
    anet = SPPaRA(net)

    assert isinstance(net.seq[0], torch.nn.Conv1d)
    assert isinstance(anet.seq[0], SPPaRAConv1d)  # Different module
    assert isinstance(net.fc2, torch.nn.Linear)
    assert isinstance(anet.fc2, torch.nn.Linear)  # Same module

    # Same weight
    assert torch.equal(net.seq[0].weight, anet.seq[0].weight)

    # Adapted weight close but not equal
    assert torch.allclose(
        net.seq[0].weight, anet.seq[0].adapted_weight, atol=_DEFAULT_INIT_RANGE + EPS
    )
    assert not torch.equal(net.seq[0].weight, anet.seq[0].adapted_weight)
