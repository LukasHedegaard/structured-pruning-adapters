from typing import Sequence

import torch
import torch_pruning as tp
from torch import nn
from torch_pruning.pruner import function

from .splora import (
    SPLoRAConv1d,
    SPLoRAConv2d,
    SPLoRAConv3d,
    SPLoRALinear,
    SPLoRAMultiheadAttention,
    _SPLoRAConvNd,
)

__all__ = [
    "SPLoRALinearPruner",
    "SPLoRAConvPruner",
    "customized_pruners",
    "root_module_types",
    "MagnitudeImportance",
]


class SPLoRALinearPruner(function.BasePruningFunc):
    TARGET_MODULES = SPLoRALinear

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()
        layer.out_features = layer.out_features - len(idxs)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        layer.adapter.out_features = layer.adapter.out_features - len(idxs)
        layer.adapter.cols = self._prune_parameter_and_grad(
            layer.adapter.cols, keep_idxs, 1
        )
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()
        layer.in_features = layer.in_features - len(idxs)

        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        layer.adapter.in_features = layer.adapter.in_features - len(idxs)
        layer.adapter.rows = self._prune_parameter_and_grad(
            layer.adapter.rows, keep_idxs, 2
        )
        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features


splora_linear_pruner = SPLoRALinearPruner()


class SPLoRAConvPruner(function.BasePruningFunc):
    TARGET_MODULE = _SPLoRAConvNd

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels = layer.out_channels - len(idxs)
        if not layer.transposed:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        else:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)

        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)

        layer.adapter.out_features = layer.adapter.out_features - len(idxs)
        layer.adapter.cols = self._prune_parameter_and_grad(
            layer.adapter.cols, keep_idxs, 1
        )
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        layer.in_channels = layer.in_channels - len(idxs)
        if layer.groups > 1:
            keep_idxs = keep_idxs[: len(keep_idxs) // layer.groups]

        if not layer.transposed:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        else:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)

        layer.adapter.in_features = layer.adapter.in_features - len(idxs)
        layer.adapter.rows = self._prune_parameter_and_grad(
            layer.adapter.rows, keep_idxs, 2
        )
        return layer

    def get_out_channels(self, layer):
        return layer.out_channels

    def get_in_channels(self, layer):
        return layer.in_channels


splora_conv_pruner = SPLoRAConvPruner()


class SPLoRAMultiheadAttentionPruner(function.BasePruningFunc):
    TARGET_MODULES = SPLoRAMultiheadAttention

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        assert (layer.embed_dim - len(idxs)) % layer.num_heads == 0, (
            "embed_dim (%d) of MultiheadAttention after pruning must divide evenly by `num_heads` (%d)"
            % (layer.embed_dim, layer.num_heads)
        )

    def prune_out_channels(
        self, layer: SPLoRAMultiheadAttention, idxs: Sequence[int]
    ) -> nn.Module:
        """
        Note: The `torch_pruning.pruning.function.MultiheadAttentionPruner`
        implementation is fundamentally flawed.

        There is no good reason to prune other parameters than `out_proj` besides to
        satisfy that `embed_dim` is consistent across all its uses.
        Currently, however, the PyTorch implementation of nn.MultiheadAttention and
        `F.multi_head_attention_forward` don't let us individually modify the dims of
        different (valid) uses of `embed_dim`.
        See https://github.com/pytorch/pytorch/issues/103668

        The pruning that is currently happening can be considered random for all but the
        `out_proj` output dimensionality. To keep in line with the torch_pruning impl,
        we'll (reluctantly) use the same strategy here.
        """
        keep_idxs = list(set(range(layer.embed_dim)) - set(idxs))
        keep_idxs.sort()
        pruning_idxs_repeated = (
            idxs
            + [i + layer.embed_dim for i in idxs]
            + [i + 2 * layer.embed_dim for i in idxs]
        )
        keep_idxs_3x_repeated = list(
            set(range(3 * layer.embed_dim)) - set(pruning_idxs_repeated)
        )
        keep_idxs_3x_repeated.sort()

        layer.q_proj = splora_linear_pruner.prune_out_channels(layer.q_proj, idxs)
        layer.q_proj = splora_linear_pruner.prune_in_channels(layer.q_proj, idxs)

        layer.k_proj = splora_linear_pruner.prune_out_channels(layer.k_proj, idxs)
        layer.k_proj = splora_linear_pruner.prune_in_channels(layer.k_proj, idxs)

        layer.v_proj = splora_linear_pruner.prune_out_channels(layer.v_proj, idxs)
        layer.v_proj = splora_linear_pruner.prune_in_channels(layer.v_proj, idxs)

        if layer.in_proj_bias is not None:
            layer.in_proj_bias = self._prune_parameter_and_grad(
                layer.in_proj_bias, keep_idxs_3x_repeated, 0
            )
        if layer.bias_k is not None:
            layer.bias_k = self._prune_parameter_and_grad(layer.bias_k, keep_idxs, 2)
        if layer.bias_v is not None:
            layer.bias_v = self._prune_parameter_and_grad(layer.bias_v, keep_idxs, 2)

        layer.out_proj = splora_linear_pruner.prune_out_channels(layer.out_proj, idxs)
        layer.out_proj = splora_linear_pruner.prune_in_channels(layer.out_proj, idxs)

        layer.embed_dim = layer.embed_dim - len(idxs)
        layer.head_dim = layer.embed_dim // layer.num_heads
        layer.kdim = layer.embed_dim
        layer.vdim = layer.embed_dim
        return layer

    def get_out_channels(self, layer: SPLoRAMultiheadAttention):
        return layer.embed_dim

    prune_in_channels = prune_out_channels
    get_in_channels = get_out_channels


splora_mha_pruner = SPLoRAMultiheadAttentionPruner()

# Pass this dict to the "customized_pruners" argument of pruners in the Torch Pruning lib
customized_pruners = {
    SPLoRALinear: splora_linear_pruner,
    _SPLoRAConvNd: splora_conv_pruner,
    SPLoRAConv1d: splora_conv_pruner,
    SPLoRAConv2d: splora_conv_pruner,
    SPLoRAConv3d: splora_conv_pruner,
    SPLoRAMultiheadAttention: splora_mha_pruner,
}

# Pass this dict to the "root_module_types" argument of pruners in the Torch Pruning lib
root_module_types = [
    nn.modules.conv._ConvNd,
    nn.Linear,
    nn.LSTM,
    SPLoRALinear,
    _SPLoRAConvNd,
    SPLoRAConv1d,
    SPLoRAConv2d,
    SPLoRAConv3d,
    SPLoRAMultiheadAttention,
]


class MagnitudeImportance(tp.importance.MagnitudeImportance):
    # Near identical implementation
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):  # noqa: C901
        group_imp = []
        # Get group norm
        # print(group.details())
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)

            # SPLoRA out_channels
            elif prune_fn in [  # Added
                splora_linear_pruner.prune_out_channels,
                splora_conv_pruner.prune_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.adapted_weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.adapted_weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)

            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                # is_conv_flatten_linear = False
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                if (
                    ch_groups > 1
                    and prune_fn == function.prune_conv_in_channels
                    and layer.groups == 1
                ):
                    # non-grouped conv and group convs
                    w = (
                        w.view(
                            w.shape[0] // group_imp[0].shape[0],
                            group_imp[0].shape[0],
                            w.shape[1],
                        )
                        .transpose(0, 1)
                        .flatten(1)
                    )
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    if len(local_norm) == len(group_imp[0]):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # SPLoRA in_channels
            elif prune_fn in [  # Added
                splora_linear_pruner.prune_in_channels,
                splora_conv_pruner.prune_in_channels,
            ]:
                # is_conv_flatten_linear = False
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.adapted_weight).flatten(1)
                else:
                    w = (layer.adapted_weight).transpose(0, 1).flatten(1)
                if (
                    ch_groups > 1
                    and prune_fn == function.prune_conv_in_channels
                    and layer.groups == 1
                ):
                    # non-grouped conv and group convs
                    w = (
                        w.view(
                            w.shape[0] // group_imp[0].shape[0],
                            group_imp[0].shape[0],
                            w.shape[1],
                        )
                        .transpose(0, 1)
                        .flatten(1)
                    )
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    if len(local_norm) == len(group_imp[0]):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups > 1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    # print(local_norm.shape)
                    group_imp.append(local_norm)
        if len(group_imp) == 0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) == imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
