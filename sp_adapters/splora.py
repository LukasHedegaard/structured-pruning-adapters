from logging import getLogger
from typing import Iterator, Tuple, Union

import torch
from torch import nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .base import AdaptableModule
from .lora import LowRankMatrix
from .utils import copy_linear_params_, recursive_replace

logger = getLogger(__name__)

_DEFAULT_RANK = 16
_DEFAULT_INIT_RANGE = 1e-4


def _configure_parameter_read(
    self,
    adapter_weights_only=True,
    in_features_mask: torch.BoolTensor = None,
    out_features_mask: torch.BoolTensor = None,
):
    self._read_adapter_weights_only = adapter_weights_only
    self._in_features_mask = in_features_mask
    self._out_features_mask = out_features_mask


def _named_parameters(
    self, prefix: str = "", recurse: bool = True
) -> Iterator[Tuple[str, nn.Parameter]]:
    for name, param in nn.Module.named_parameters(self, prefix, recurse):
        if not self._read_adapter_weights_only or "adapter" in name:
            if name == "adapter.rows" and self._in_features_mask is not None:
                param = param[:, :, self._in_features_mask].flatten()
            if name == "adapter.cols" and self._out_features_mask is not None:
                param = param[:, self._out_features_mask, :].flatten()
            yield (name, param)


class SPLoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
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
            1, in_features, out_features, rank, init_range=init_range
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
        if self.weight.requires_grad:
            self.weight.requires_grad = False
            logger.warning("Forcing `weight.requires_grad = False`")
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
        return _configure_parameter_read(
            self, adapter_weights_only, in_features_mask, out_features_mask
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        return _named_parameters(self, prefix, recurse)

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
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


class _SPLoRAConvNd:
    def __init__(
        self,
        ConvCls: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
    ):
        self._nd = int(ConvCls.__name__[-2])
        self._ConvCls = ConvCls
        self.adapter = LowRankMatrix(
            1, in_channels, out_channels, rank, init_range=init_range
        )
        if bias:
            self.adapter_bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
            nn.init.uniform_(self.adapter_bias, -init_range, init_range)
        else:
            self.register_parameter("adapter_bias", None)

        self.configure_parameter_read()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.adapted_weight, self.adapted_bias)

    @property
    def adapted_weight(self) -> nn.Parameter:
        assert not self.weight.requires_grad
        w_diag = torch.zeros_like(self.weight)
        kdx = self.kernel_size[0] // 2
        center_inds = [kdx for _ in range(self._nd)]
        # w_diag[:, :, *center_inds] += self.adapter().squeeze(0) # Python 3.11+
        wdx = [slice(None), slice(None)] + center_inds
        w_diag.__setitem__(wdx, w_diag.__getitem__(wdx) + self.adapter().squeeze(0))
        return self.weight + w_diag

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
        return _configure_parameter_read(
            self, adapter_weights_only, in_features_mask, out_features_mask
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        return _named_parameters(self, prefix, recurse)

    def to_module(self) -> torch.nn.modules.conv._ConvNd:
        instance = self._ConvCls(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
            self.weight.device,
            self.weight.dtype,
        )
        instance.weight = torch.nn.Parameter(self.adapted_weight)
        if self.bias is not None:
            instance.bias = torch.nn.Parameter(self.adapted_bias)
        return instance


class SPLoRAConv1d(_SPLoRAConvNd, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
    ):
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert len(set(self.kernel_size)) == 1, "All dimensions of kernel_size equal."
        assert kernel_size[0] % 2 == 1, "kernel_size should be odd."
        _SPLoRAConvNd.__init__(
            self,
            nn.Conv1d,
            in_channels,
            out_channels,
            bias,
            rank,
            init_range,
            device,
            dtype,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPLoRAConv1d":
        instance = SPLoRAConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            rank,
            init_range,
            module.weight.device,
            module.weight.dtype,
        )
        copy_linear_params_(module, instance)
        return instance


class SPLoRAConv2d(_SPLoRAConvNd, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert len(set(self.kernel_size)) == 1, "All dimensions of kernel_size equal."
        assert kernel_size[0] % 2 == 1, "kernel_size should be odd."
        _SPLoRAConvNd.__init__(
            self,
            nn.Conv2d,
            in_channels,
            out_channels,
            bias,
            rank,
            init_range,
            device,
            dtype,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPLoRAConv2d":
        instance = SPLoRAConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            rank,
            init_range,
            module.weight.device,
            module.weight.dtype,
        )
        copy_linear_params_(module, instance)
        return instance


class SPLoRAConv3d(_SPLoRAConvNd, nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
    ):
        nn.Conv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert len(set(self.kernel_size)) == 1, "All dimensions of kernel_size equal."
        assert kernel_size[0] % 2 == 1, "kernel_size should be odd."
        _SPLoRAConvNd.__init__(
            self,
            nn.Conv3d,
            in_channels,
            out_channels,
            bias,
            rank,
            init_range,
            device,
            dtype,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPLoRAConv3d":
        instance = SPLoRAConv3d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            rank,
            init_range,
            module.weight.device,
            module.weight.dtype,
        )
        copy_linear_params_(module, instance)
        return instance


def SPLoRA(
    module: AdaptableModule,
    rank: Union[
        int, float
    ] = _DEFAULT_RANK,  # rank (int) or fraction of output_channels
    init_range: float = _DEFAULT_INIT_RANGE,
    inplace=False,
    replacements=[
        (nn.Linear, SPLoRALinear),
        (nn.Conv1d, SPLoRAConv1d),
        (nn.Conv2d, SPLoRAConv2d),
        (nn.Conv3d, SPLoRAConv3d),
    ],
):
    mod = module
    for FromCls, ToSPLoRACls in replacements:
        mod = recursive_replace(
            mod,
            FromCls,
            ToSPLoRACls.from_module,
            inplace,
            None,
            rank=rank,
            init_range=init_range,
        )
    return mod
