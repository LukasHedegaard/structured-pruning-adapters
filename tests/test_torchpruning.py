# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

import sp_adapters as spa


def test_SPLoRALinear_pruning():
    class FullyConnectedNet(nn.Module):
        def __init__(self, input_size, num_classes, hidden_units):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_units)
            self.customized_layer = spa.SPLoRALinear(
                hidden_units, 2 * hidden_units, rank=2
            )
            self.fc2 = nn.Linear(2 * hidden_units, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.customized_layer(x)
            y_hat = self.fc2(x)
            return y_hat

    num_classes = 10
    d_in = 8
    d_hidden = 16
    model = FullyConnectedNet(d_in, num_classes, d_hidden)

    old_weight = model.customized_layer.weight.clone()
    old_bias = model.customized_layer.bias.clone()
    old_adapter_rows = model.customized_layer.adapter.rows.clone()
    old_adapter_cols = model.customized_layer.adapter.cols.clone()

    DG = tp.DependencyGraph()

    DG.build_dependency(
        model,
        example_inputs=torch.randn(1, d_in),
        customized_pruners=spa.torch_pruning.customized_pruners,
    )
    # Get a pruning group according to the dependency graph.
    # idxs is the indices of pruned filters.
    # Prune out channels
    prune_idxs = [0, 1, 6]
    keep_idxs = list(set(range(model.customized_layer.in_features)) - set(prune_idxs))
    pruning_group = DG.get_pruning_group(
        model.fc1, tp.prune_linear_out_channels, idxs=prune_idxs
    )
    print(pruning_group)

    # Execute this group (prune the model)
    pruning_group.prune()

    # Changed
    assert torch.allclose(old_weight[:, keep_idxs], model.customized_layer.weight)
    assert torch.allclose(
        old_adapter_rows[:, :, keep_idxs],
        model.customized_layer.adapter.rows,
    )

    assert model.fc1.out_features == d_hidden - len(prune_idxs)
    assert model.customized_layer.in_features == d_hidden - len(prune_idxs)
    assert model.customized_layer.adapter.in_features == d_hidden - len(prune_idxs)

    # Unchanged
    assert torch.allclose(old_bias, model.customized_layer.bias)
    assert torch.allclose(old_adapter_cols, model.customized_layer.adapter.cols)

    assert model.customized_layer.out_features == 2 * d_hidden
    assert model.fc2.in_features == 2 * d_hidden
    assert model.customized_layer.adapter.out_features == 2 * d_hidden

    # print("The pruned model:\n", model)
    assert model(torch.randn(1, d_in)).shape == (1, num_classes)

    # Repeat for in_features
    keep_idxs_out = list(
        set(range(model.customized_layer.out_features)) - set(prune_idxs)
    )
    pruning_group = DG.get_pruning_group(
        model.fc2, tp.prune_linear_in_channels, idxs=prune_idxs
    )

    pruning_group.prune()

    # Changed
    assert torch.allclose(
        old_weight[keep_idxs_out][:, keep_idxs], model.customized_layer.weight
    )
    assert torch.allclose(old_bias[keep_idxs_out], model.customized_layer.bias)
    assert torch.allclose(
        old_bias[keep_idxs_out],
        model.customized_layer.bias,
    )
    assert torch.allclose(
        old_adapter_cols[:, keep_idxs_out],
        model.customized_layer.adapter.cols,
    )

    assert model.customized_layer.out_features == 2 * d_hidden - len(prune_idxs)
    assert model.customized_layer.adapter.out_features == 2 * d_hidden - len(prune_idxs)
    assert model.fc2.in_features == 2 * d_hidden - len(prune_idxs)

    # print("The pruned model:\n", model)
    assert model(torch.randn(1, d_in)).shape == (1, num_classes)


def test_SPLoRALinear_magpruner():
    d_in = 8
    d_out = 10

    example_input = torch.randn(1, d_in)

    model = spa.SPLoRALinear(d_in, d_out, rank=2)

    # Initialize with known weights
    model.weight = torch.nn.Parameter(
        torch.outer(
            torch.concat(
                [
                    torch.arange(d_out // 2, dtype=torch.float),
                    torch.arange(d_out // 2, 0, step=-1, dtype=torch.float),
                ]
            ),
            torch.concat(
                [
                    torch.arange(d_in // 2, dtype=torch.float),
                    torch.arange(d_in // 2, 0, step=-1, dtype=torch.float),
                ]
            ),
        )
    )
    model.bias = torch.nn.Parameter(torch.arange(d_out, dtype=torch.float))

    # Decreasing weights
    model.adapter_bias = torch.nn.Parameter(
        2 * torch.arange(d_out, 0, step=-1, dtype=torch.float)
    )

    model.adapter.rows = torch.nn.Parameter(torch.ones_like(model.adapter.rows))
    model.adapter.rows = torch.nn.Parameter(
        torch.outer(
            torch.arange(model.adapter.rank, 0, step=-1, dtype=torch.float),
            torch.arange(d_in, 0, step=-1, dtype=torch.float),
        ).unsqueeze(0)
    )

    model.adapter.cols = torch.nn.Parameter(
        torch.outer(
            torch.arange(d_out, 0, step=-1, dtype=torch.float),
            torch.arange(model.adapter.rank, 0, step=-1, dtype=torch.float),
        ).unsqueeze(0)
    )

    old_lin_weight = model.weight.clone()
    old_bias = model.bias.clone()
    old_adapter_rows = model.adapter.rows.clone()
    old_adapter_cols = model.adapter.cols.clone()

    imp = spa.torch_pruning.MagnitudeImportance(p=2)

    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=example_input,
        importance=imp,  # Importance Estimator
        global_pruning=False,  # Please refer to Page 9 of https://www.cs.princeton.edu/courses/archive/spring21/cos598D/lectures/pruning.pdf
        ch_sparsity=0.5,  # global sparsity for all layers
        # ch_sparsity_dict = {model.conv1: 0.2}, # manually set the sparsity of model.conv1
        iterative_steps=1,  # number of steps to achieve the target ch_sparsity.
        # ignored_layers=,  # ignore final linear classifier
        round_to=None,  # round channels
        # unwrapped_parameters=[ (model.features[1][1].layer_scale, 0), (model.features[5][4].layer_scale, 0) ],
        customized_pruners=spa.torch_pruning.customized_pruners,
        root_module_types=spa.torch_pruning.root_module_types,
    )

    # Check that pruning importance group uses adapter weights
    groups = list(
        pruner.DG.get_all_groups(
            ignored_layers=pruner.ignored_layers,
            root_module_types=pruner.root_module_types,
        )
    )
    group = groups[0]  # only one group
    imp = pruner.estimate_importance(group, ch_groups=pruner.get_channel_groups(group))
    # imp = [2.4912, 2.0462, 1.6452, 1.2883, 0.9753, 0.7063, 0.4520, 0.2543, 0.1130, 0.0283]
    assert torch.argmax(imp) == 0

    # Apply pruning
    pruner.step()

    # Shapes changed
    assert model.weight.shape == (d_out // 2, d_in)
    assert model.adapter.cols.shape == (1, d_out // 2, 2)
    assert model.bias.shape == (d_out // 2,)
    assert model.adapter.rows.shape == (1, 2, d_in)  # unchanged

    # The first (most important) weights were kept
    assert torch.equal(model.weight, old_lin_weight[: d_out // 2])
    assert torch.equal(model.bias, old_bias[: d_out // 2])
    assert torch.equal(model.adapter.cols, old_adapter_cols[:, : d_out // 2])
    assert torch.equal(model.adapter.rows, old_adapter_rows)  # Unchanged


def test_SPLoRAConv_pruning():
    class FullyConnectedNet(nn.Module):
        def __init__(self, input_size, num_classes, hidden_units):
            super().__init__()
            self.c1 = nn.Conv1d(input_size, hidden_units, kernel_size=3)
            self.customized_layer = spa.SPLoRAConv1d(
                hidden_units,
                2 * hidden_units,
                kernel_size=3,
                rank=2,
            )
            self.c2 = nn.Conv1d(2 * hidden_units, num_classes, kernel_size=3)

        def forward(self, x):
            x = F.relu(self.c1(x))
            x = self.customized_layer(x)
            y_hat = self.c2(x)
            return y_hat

    num_classes = 10
    d_in = 8
    d_hidden = 16
    seq_len = 7
    model = FullyConnectedNet(d_in, num_classes, d_hidden)

    old_weight = model.customized_layer.weight.clone()
    old_bias = model.customized_layer.bias.clone()
    old_adapter_rows = model.customized_layer.adapter.rows.clone()
    old_adapter_cols = model.customized_layer.adapter.cols.clone()

    DG = tp.DependencyGraph()

    DG.build_dependency(
        model,
        example_inputs=torch.randn(1, d_in, seq_len),
        customized_pruners=spa.torch_pruning.customized_pruners,
    )
    # Get a pruning group according to the dependency graph.
    # idxs is the indices of pruned filters.
    # Prune out channels
    prune_idxs = [0, 1, 6]
    keep_idxs = list(set(range(model.customized_layer.in_channels)) - set(prune_idxs))
    pruning_group = DG.get_pruning_group(
        model.c1, tp.prune_conv_out_channels, idxs=prune_idxs
    )
    print(pruning_group)

    # Execute this group (prune the model)
    pruning_group.prune()

    # Changed
    assert torch.allclose(old_weight[:, keep_idxs], model.customized_layer.weight)
    assert torch.allclose(
        old_adapter_rows[:, :, keep_idxs],
        model.customized_layer.adapter.rows,
    )

    assert model.c1.out_channels == d_hidden - len(prune_idxs)
    assert model.customized_layer.in_channels == d_hidden - len(prune_idxs)
    assert model.customized_layer.adapter.in_features == d_hidden - len(prune_idxs)

    # Unchanged
    assert torch.allclose(old_bias, model.customized_layer.bias)
    assert torch.allclose(old_adapter_cols, model.customized_layer.adapter.cols)

    assert model.customized_layer.out_channels == 2 * d_hidden
    assert model.c2.in_channels == 2 * d_hidden
    assert model.customized_layer.adapter.out_features == 2 * d_hidden

    # print("The pruned model:\n", model)
    assert model(torch.randn(1, d_in, seq_len)).shape == (1, num_classes, 1)

    # Repeat for in_channels
    keep_idxs_out = list(
        set(range(model.customized_layer.out_channels)) - set(prune_idxs)
    )
    pruning_group = DG.get_pruning_group(
        model.c2, tp.prune_conv_in_channels, idxs=prune_idxs
    )

    pruning_group.prune()

    # Changed
    assert torch.allclose(
        old_weight[keep_idxs_out][:, keep_idxs], model.customized_layer.weight
    )
    assert torch.allclose(old_bias[keep_idxs_out], model.customized_layer.bias)
    assert torch.allclose(
        old_bias[keep_idxs_out],
        model.customized_layer.bias,
    )
    assert torch.allclose(
        old_adapter_cols[:, keep_idxs_out],
        model.customized_layer.adapter.cols,
    )

    assert model.customized_layer.out_channels == 2 * d_hidden - len(prune_idxs)
    assert model.customized_layer.adapter.out_features == 2 * d_hidden - len(prune_idxs)
    assert model.c2.in_channels == 2 * d_hidden - len(prune_idxs)

    # print("The pruned model:\n", model)
    assert model(torch.randn(1, d_in, seq_len)).shape == (1, num_classes, 1)


def test_SPLoRAMultiheadAttention_pruning():
    class MhaNet(nn.Module):
        def __init__(self, input_size, num_classes, hidden_units):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_units)
            # self.mha = nn.MultiheadAttention(hidden_units, num_heads=4)
            self.mha = spa.SPLoRAMultiheadAttention(hidden_units, num_heads=4, rank=2)
            self.fc2 = nn.Linear(hidden_units, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x, _ = self.mha(x, x, x)
            y_hat = self.fc2(x)
            return y_hat

    num_classes = 3
    d_in = 8
    d_hidden = 16
    seq_len = 10
    model = MhaNet(d_in, num_classes, d_hidden)
    example_input = torch.randn(seq_len, d_in)

    DG = tp.DependencyGraph()

    DG.build_dependency(
        model,
        example_inputs=example_input,
        customized_pruners=spa.torch_pruning.customized_pruners,
    )
    # Get a pruning group according to the dependency graph.
    # idxs is the indices of pruned filters.
    # Prune out channels
    prune_idxs = [0, 1, 3, 6]
    pruning_group = DG.get_pruning_group(
        model.fc1, tp.prune_linear_out_channels, idxs=prune_idxs
    )
    print(pruning_group)

    # Execute this group (prune the model)
    pruning_group.prune()

    # Check shapes
    embed_dim = d_hidden - len(prune_idxs)
    assert model.mha.q_proj.adapted_weight.shape == (embed_dim, embed_dim)
    assert model.mha.k_proj.adapted_weight.shape == (embed_dim, embed_dim)
    assert model.mha.v_proj.adapted_weight.shape == (embed_dim, embed_dim)
    assert model.mha.out_proj.adapted_weight.shape == (embed_dim, embed_dim)


if __name__ == "__main__":
    # test_SPLoRALinear_pruning()
    test_SPLoRALinear_magpruner()
