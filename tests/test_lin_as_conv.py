import time

import torch
from torch import nn


def apply_lin_as_pointwise_conv(
    input: torch.Tensor,
    lin: torch.nn.Linear,
):
    batch_size, in_features, *rest = input.shape

    return (
        lin(input.view(batch_size, in_features, -1).transpose(-2, -1))
        .transpose(-1, -2)
        .view(batch_size, lin.out_features, *rest)
    )


def test_lin_as_conv():
    # This test is a general sanity check, reaffirming
    # that we can indeed use linear as pointwise convolution
    batch_size = 2
    in_features = 32
    out_features = 64
    num_runs = 1

    print("======= 1D case =======")
    seq_len = 128

    input = torch.rand(batch_size, in_features, seq_len)

    c1 = nn.Conv1d(in_features, out_features, kernel_size=1, padding=0, bias=False)
    start_time = time.time()
    for _ in range(num_runs):
        output_conv = c1(input)
    print("c1 time", time.time() - start_time)

    l1 = nn.Linear(in_features, out_features, bias=False)
    l1.weight.data[:] = c1.weight.data.view(out_features, in_features)
    start_time = time.time()
    for _ in range(num_runs):
        output_lin = apply_lin_as_pointwise_conv(input, l1)
    print("l1 time", time.time() - start_time)
    max_err = torch.max(output_conv - output_lin)
    print("Maximal error: {}".format(max_err))
    assert max_err < 1e-9

    print("======= 2D case =======")
    h, w = 128, 128
    input = torch.rand(batch_size, in_features, h, w)

    c2 = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=False)
    start_time = time.time()
    for _ in range(num_runs):
        output_conv = c2(input)
    print("c2 time", time.time() - start_time)

    l2 = nn.Linear(in_features, out_features, bias=False)
    l2.weight.data[:] = c2.weight.data.view(out_features, in_features)
    start_time = time.time()
    for _ in range(num_runs):
        output_lin = apply_lin_as_pointwise_conv(input, l2)
    print("l2 time", time.time() - start_time)
    max_err = torch.max(output_conv - output_lin)
    print("Maximal error: {}".format(max_err))
    assert max_err < 1e-9

    print("======= 3D case =======")
    h, w, t = 64, 64, 16
    input = torch.rand(batch_size, in_features, h, w, t)

    c3 = nn.Conv3d(in_features, out_features, kernel_size=1, padding=0, bias=False)
    start_time = time.time()
    for _ in range(num_runs):
        output_conv = c3(input)
    print("c3 time", time.time() - start_time)

    l3 = nn.Linear(in_features, out_features, bias=False)
    l3.weight.data[:] = c3.weight.data.view(out_features, in_features)
    start_time = time.time()
    for _ in range(num_runs):
        output_lin = apply_lin_as_pointwise_conv(input, l3)
    print("l3 time", time.time() - start_time)
    max_err = torch.max(output_conv - output_lin)
    print("Maximal error: {}".format(max_err))
    assert max_err < 1e-9
