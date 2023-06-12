from typing import Sequence

import torch
import torch_pruning as tp
from torch import nn
from torch_pruning.pruner import function

from .splora import SPLoRALinear  # , _SPLoRAConvNd

__all__ = [
    "SPLoRALinearPruner",
    # "SPLoRAConvPruner",
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


# Pass this dict to the "customized_pruners" argument of pruners in the Torch Pruning lib
splora_linear_pruner = SPLoRALinearPruner()
customized_pruners = {
    SPLoRALinear: splora_linear_pruner,
    # _SPLoRAConvNd: SPLoRAConvPruner,
}

root_module_types = [nn.modules.conv._ConvNd, nn.Linear, nn.LSTM, SPLoRALinear]


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
            elif prune_fn in [
                splora_linear_pruner.prune_in_channels,  # Added
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
