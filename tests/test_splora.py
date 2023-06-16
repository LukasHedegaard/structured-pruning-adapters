import torch
from torch import nn

from sp_adapters import SPLoRA
from sp_adapters.splora import (
    _DEFAULT_INIT_RANGE,
    SPLoRALinear,
    named_parameters,
    parameters,
)

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

    # Adapter params changed!
    assert not torch.equal(lin.bias, splin.bias)
    assert not torch.equal(splin.adapter.cols, prev_adapter_cols)
    assert not torch.equal(splin.adapter.rows, prev_adapter_rows)

    # Count params
    # tot_params = sum([torch.Tensor([p.shape]).prod() for p in splin.parameters()])

    # Count masked params
    in_features_mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.bool)
    out_features_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    tot_masked_params = sum(
        [
            torch.Tensor([p.shape]).prod()
            for p in parameters(
                splin,
                adapter_weights_only=True,
                in_features_mask=in_features_mask,
                out_features_mask=out_features_mask,
            )
        ]
    )
    #                           in_feat    out_feat   bias
    assert tot_masked_params == rank * 4 + rank * 2 + 2

    # Export to lin
    lin2 = splin.to_module()
    assert torch.equal(lin2.weight, splin.adapted_weight)
    assert torch.equal(lin2.bias, splin.bias)


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
        assert not torch.equal(spconv.adapter.cols, prev_adapter_cols)
        assert not torch.equal(spconv.adapter.rows, prev_adapter_rows)

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
            assert tot_masked_params == 4 * rank + 2 * rank + 2
        else:
            assert tot_masked_params == 4 * rank + 2 * rank

        # Export to conv
        conv2 = spconv.to_module()
        assert torch.equal(conv2.weight, spconv.adapted_weight)
        if bias:
            assert torch.equal(conv2.bias, spconv.bias)


def test_splora_mha_equal_qkv_dim():
    rank = 2
    embed_dim = 16
    num_heads = 4
    seq_len = 8
    x = torch.randn(seq_len, embed_dim)
    y = torch.randn(seq_len, embed_dim)

    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
    )

    # Transfer
    spmha = SPLoRA(mha, rank)

    # Weights are all close, biases are equal
    assert torch.equal(
        mha.in_proj_weight,
        torch.cat(
            (
                spmha.q_proj.weight,
                spmha.k_proj.weight,
                spmha.v_proj.weight,
            )
        ),
    )
    assert torch.allclose(  # in_proj_weight property uses adapted_weights
        mha.in_proj_weight, spmha.in_proj_weight, atol=_DEFAULT_INIT_RANGE
    )
    assert torch.allclose(
        mha.out_proj.weight, spmha.out_proj.adapted_weight, atol=_DEFAULT_INIT_RANGE
    )

    # Prepare optim
    optimizer = torch.optim.Adam(spmha.parameters())
    optimizer.zero_grad()

    # Save for later comparison
    prev_q_proj_cols = torch.clone(spmha.q_proj.adapter.cols)
    prev_q_proj_rows = torch.clone(spmha.q_proj.adapter.rows)
    prev_k_proj_cols = torch.clone(spmha.k_proj.adapter.cols)
    prev_k_proj_rows = torch.clone(spmha.k_proj.adapter.rows)
    prev_v_proj_cols = torch.clone(spmha.v_proj.adapter.cols)
    prev_v_proj_rows = torch.clone(spmha.v_proj.adapter.rows)
    prev_out_proj_cols = torch.clone(spmha.out_proj.adapter.cols)
    prev_out_proj_rows = torch.clone(spmha.out_proj.adapter.rows)

    # Forwards are approx equal since adapters were initialized with near-zero values
    mha_out, _ = mha.forward(x, x, x)
    spmha_out, _ = spmha.forward(x, x, x)
    assert torch.allclose(mha_out, spmha_out, atol=1e-3)

    # Train a bit
    mse = nn.MSELoss()
    loss = mse(spmha_out, y)
    loss.backward()
    optimizer.step()

    # MHA weight params remain unchanged
    assert torch.equal(
        mha.in_proj_weight,
        torch.cat(
            (
                spmha.q_proj.weight,
                spmha.k_proj.weight,
                spmha.v_proj.weight,
            )
        ),
    )
    assert torch.equal(
        mha.out_proj.weight,
        spmha.out_proj.weight,
    )

    # Adapter params changed
    assert not torch.equal(prev_q_proj_cols, spmha.q_proj.adapter.cols)
    assert not torch.equal(prev_q_proj_rows, spmha.q_proj.adapter.rows)
    assert not torch.equal(prev_k_proj_cols, spmha.k_proj.adapter.cols)
    assert not torch.equal(prev_k_proj_rows, spmha.k_proj.adapter.rows)
    assert not torch.equal(prev_v_proj_cols, spmha.v_proj.adapter.cols)
    assert not torch.equal(prev_v_proj_rows, spmha.v_proj.adapter.rows)
    assert not torch.equal(prev_out_proj_cols, spmha.out_proj.adapter.cols)
    assert not torch.equal(prev_out_proj_rows, spmha.out_proj.adapter.rows)

    # Export to mha
    mha2 = spmha.to_module()

    # Weights and biases are equal
    assert torch.equal(  # in_proj_weight property uses adapted_weights
        mha2.in_proj_weight, spmha.in_proj_weight
    )
    assert torch.equal(mha2.out_proj.weight, spmha.out_proj.adapted_weight)


def test_splora_mha_diff_qkv_dim():
    rank = 2
    embed_dim = 16
    kdim = vdim = 24
    num_heads = 4
    seq_len = 8
    batch_size = 2
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, kdim)
    v = torch.randn(batch_size, seq_len, vdim)
    y = torch.randn(batch_size, seq_len, embed_dim)

    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=True,
        add_zero_attn=False,
        kdim=kdim,
        vdim=vdim,
        batch_first=True,
    )
    mha.in_proj_bias = torch.nn.Parameter(torch.rand_like(mha.in_proj_bias))

    # Transfer
    spmha = SPLoRA(mha, rank)

    # Weights are all close, biases are equal

    assert torch.equal(mha.q_proj_weight, spmha.q_proj.weight)
    assert torch.equal(mha.k_proj_weight, spmha.k_proj.weight)
    assert torch.equal(mha.v_proj_weight, spmha.v_proj.weight)
    assert torch.equal(mha.out_proj.weight, spmha.out_proj.weight)
    assert torch.allclose(
        mha.q_proj_weight, spmha.q_proj.adapted_weight, atol=_DEFAULT_INIT_RANGE
    )
    assert torch.allclose(
        mha.k_proj_weight, spmha.k_proj.adapted_weight, atol=_DEFAULT_INIT_RANGE
    )
    assert torch.allclose(
        mha.v_proj_weight, spmha.v_proj.adapted_weight, atol=_DEFAULT_INIT_RANGE
    )
    assert torch.allclose(
        mha.out_proj.weight, spmha.out_proj.adapted_weight, atol=_DEFAULT_INIT_RANGE
    )

    assert torch.equal(mha.in_proj_bias, spmha.in_proj_bias)
    assert torch.equal(mha.bias_k, spmha.bias_k)
    assert torch.equal(mha.bias_v, spmha.bias_v)
    assert torch.equal(mha.out_proj.bias, spmha.out_proj.bias)

    # Prepare optim
    optimizer = torch.optim.Adam(spmha.parameters())
    optimizer.zero_grad()

    # Save for later comparison
    prev_q_proj_cols = torch.clone(spmha.q_proj.adapter.cols)
    prev_q_proj_rows = torch.clone(spmha.q_proj.adapter.rows)
    prev_k_proj_cols = torch.clone(spmha.k_proj.adapter.cols)
    prev_k_proj_rows = torch.clone(spmha.k_proj.adapter.rows)
    prev_v_proj_cols = torch.clone(spmha.v_proj.adapter.cols)
    prev_v_proj_rows = torch.clone(spmha.v_proj.adapter.rows)
    prev_out_proj_cols = torch.clone(spmha.out_proj.adapter.cols)
    prev_out_proj_rows = torch.clone(spmha.out_proj.adapter.rows)

    # Forwards are approx equal since adapters were initialized with near-zero values
    mha_out, _ = mha.forward(q, k, v)
    spmha_out, _ = spmha.forward(q, k, v)
    assert torch.allclose(mha_out, spmha_out, atol=1e-3)

    # Train a bit
    mse = nn.MSELoss()
    loss = mse(spmha_out, y)
    loss.backward()
    optimizer.step()

    # MHA weight params remain unchanged
    assert torch.equal(mha.q_proj_weight, spmha.q_proj.weight)
    assert torch.equal(mha.k_proj_weight, spmha.k_proj.weight)
    assert torch.equal(mha.v_proj_weight, spmha.v_proj.weight)
    assert torch.equal(mha.out_proj.weight, spmha.out_proj.weight)

    # Biases changed
    assert not torch.equal(mha.in_proj_bias, spmha.in_proj_bias)
    assert not torch.equal(mha.bias_k, spmha.bias_k)
    assert not torch.equal(mha.bias_v, spmha.bias_v)
    assert not torch.equal(mha.out_proj.bias, spmha.out_proj.bias)

    # Adapter params changed
    assert not torch.equal(prev_q_proj_cols, spmha.q_proj.adapter.cols)
    assert not torch.equal(prev_q_proj_rows, spmha.q_proj.adapter.rows)
    assert not torch.equal(prev_k_proj_cols, spmha.k_proj.adapter.cols)
    assert not torch.equal(prev_k_proj_rows, spmha.k_proj.adapter.rows)
    assert not torch.equal(prev_v_proj_cols, spmha.v_proj.adapter.cols)
    assert not torch.equal(prev_v_proj_rows, spmha.v_proj.adapter.rows)
    assert not torch.equal(prev_out_proj_cols, spmha.out_proj.adapter.cols)
    assert not torch.equal(prev_out_proj_rows, spmha.out_proj.adapter.rows)

    # Export to mha
    mha2 = spmha.to_module()

    # Weights and biases are equal
    assert torch.equal(mha2.q_proj_weight, spmha.q_proj.adapted_weight)
    assert torch.equal(mha2.k_proj_weight, spmha.k_proj.adapted_weight)
    assert torch.equal(mha2.v_proj_weight, spmha.v_proj.adapted_weight)
    assert torch.equal(mha2.out_proj.weight, spmha.out_proj.adapted_weight)

    assert torch.equal(mha2.in_proj_bias, spmha.in_proj_bias)
    assert torch.equal(mha2.bias_k, spmha.bias_k)
    assert torch.equal(mha2.bias_v, spmha.bias_v)
    assert torch.equal(mha2.out_proj.bias, spmha.out_proj.bias)


def test_splora_named_parameters():
    in_features = 9
    out_features = 4
    rank = 2

    splin1 = SPLoRALinear(in_features, out_features, rank=rank, bias=True)
    nps1 = {v[0] for v in named_parameters(splin1)}
    assert "adapter.rows" in nps1
    assert "adapter.cols" in nps1

    # Linear params exis, but do not show in named_parameters
    assert "weight" not in nps1
    assert isinstance(splin1.weight, nn.Parameter)
    assert isinstance(splin1.bias, nn.Parameter)

    splin2 = SPLoRALinear(in_features, out_features, rank=rank, bias=False)
    nps2 = {v[0] for v in named_parameters(splin2)}
    assert "adapter.rows" in nps2
    assert "adapter.cols" in nps2

    # Linear params exist, but do not show in named_parameters
    assert "weight" not in nps2
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

    # Adapted weight close but not equal
    assert torch.allclose(
        net.seq[0].weight, anet.seq[0].adapted_weight, atol=_DEFAULT_INIT_RANGE + EPS
    )
    assert not torch.equal(net.seq[0].weight, anet.seq[0].adapted_weight)
