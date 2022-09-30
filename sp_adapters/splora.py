from typing import Iterator, Tuple

import torch
from torch import nn

from .base import AdaptableModule
from .lora import LowRankMatrix
from .utils import copy_linear_params_, recursive_replace

_DEFAULT_INIT_RANGE = 1e-4


def SPLoRA(
    module: AdaptableModule,
    rank: int = 16,
    init_range: float = _DEFAULT_INIT_RANGE,
    inplace=False,
):
    return recursive_replace(
        module,
        nn.Linear,
        SPLoRALinear.from_module,
        inplace,
        None,
        rank=rank,
        init_range=init_range,
    )


class SPLoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 16,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self.adapter = LowRankMatrix(
            1, out_features, in_features, rank, init_near_zero=True
        )
        if bias:
            self.adapter_bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
            nn.init.uniform_(self.adapter_bias, -init_range, init_range)
        else:
            self.register_parameter("adapter_bias", None)
        self.reset_parameters()
        self.configure_parameter_read()

    def forward(self, input):
        return nn.functional.linear(input, self.adapted_weight, self.adapted_bias)

    @property
    def adapted_weight(self) -> nn.Parameter:
        assert not self.weight.requires_grad
        return self.adapter().squeeze(0) + self.weight

    @property
    def adapted_bias(self) -> nn.Parameter:
        result = None
        if self.adapter_bias is not None:
            result = self.bias + self.adapter_bias
        return result

    def configure_parameter_read(
        self,
        adapter_weights_only=True,
        in_features_mask: torch.BoolTensor = None,
        out_features_mask: torch.BoolTensor = None,
    ):
        self._read_adapter_weights_only = adapter_weights_only
        self._in_features_mask = in_features_mask
        self._out_features_mask = out_features_mask

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in super().named_parameters(prefix, recurse):
            if not self._read_adapter_weights_only or "adapter" in name:
                if name == "adapter.rows" and self._in_features_mask is not None:
                    param = param[:, :, self._in_features_mask].flatten()
                if name == "adapter.cols" and self._out_features_mask is not None:
                    param = param[:, self._out_features_mask, :].flatten()
                yield (name, param)

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        rank: int = 16,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPLoRALinear":
        instance = SPLoRALinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            rank=rank,
            init_range=init_range,
        )
        copy_linear_params_(module, instance)
        return instance

    def to_module(self) -> nn.Linear:
        instance = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        instance.weight = torch.nn.Parameter(self.adapted_weight)
        if self.bias is not None:
            instance.bias = torch.nn.Parameter(self.adapted_bias)
        return instance
