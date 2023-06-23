from logging import getLogger
from typing import Iterator, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .base import AdaptableModule
from .utils import copy_linear_params_, recursive_replace

logger = getLogger(__name__)

_DEFAULT_INIT_RANGE = 1e-4


class _SPPaRAConvNd:
    """Base class for Convolutional {S}tructured {P}runing {Pa}rallel {Re}sidual {A}dapters"""

    def __init__(
        self,
        ConvCls: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        self._nd = int(ConvCls.__name__[-2])
        self._ConvCls = ConvCls
        self.adapter = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, device=device, dtype=dtype)
        )
        nn.init.uniform_(self.adapter, -init_range, init_range)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.adapted_weight, self.bias)

    @property
    def adapted_weight(self) -> nn.Parameter:
        if self.weight.requires_grad:
            self.weight.requires_grad = False
        w_diag = torch.zeros_like(self.weight)
        kdx = self.kernel_size[0] // 2
        center_inds = [kdx for _ in range(self._nd)]
        # w_diag[:, :, *center_inds] += self.adapter # Python 3.11+
        wdx = [slice(None), slice(None)] + center_inds
        w_diag.__setitem__(wdx, w_diag.__getitem__(wdx) + self.adapter)
        return self.weight + w_diag

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
            instance.bias = torch.nn.Parameter(self.bias)
        return instance


class SPPaRAConv1d(_SPPaRAConvNd, nn.Conv1d):
    """{S}tructured {P}runing {Pa}rallel {Re}sidual {A}dapter for 1D Convolutions"""

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
        assert self.kernel_size[0] % 2 == 1, "kernel_size should be odd."
        _SPPaRAConvNd.__init__(
            self,
            nn.Conv1d,
            in_channels,
            out_channels,
            groups,
            init_range,
            device,
            dtype,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPPaRAConv1d":
        instance = SPPaRAConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            init_range,
            module.weight.device,
            module.weight.dtype,
        )
        copy_linear_params_(module, instance)
        return instance


class SPPaRAConv2d(_SPPaRAConvNd, nn.Conv2d):
    """{S}tructured {P}runing {Pa}rallel {Re}sidual {A}dapter for 2D Convolutions"""

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
        assert self.kernel_size[0] % 2 == 1, "kernel_size should be odd."
        _SPPaRAConvNd.__init__(
            self,
            nn.Conv2d,
            in_channels,
            out_channels,
            groups,
            init_range,
            device,
            dtype,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPPaRAConv2d":
        instance = SPPaRAConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            init_range,
            module.weight.device,
            module.weight.dtype,
        )
        copy_linear_params_(module, instance)
        return instance


class SPPaRAConv3d(_SPPaRAConvNd, nn.Conv3d):
    """{S}tructured {P}runing {Pa}rallel {Re}sidual {A}dapter for 3D Convolutions"""

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
        assert self.kernel_size[0] % 2 == 1, "kernel_size should be odd."
        _SPPaRAConvNd.__init__(
            self,
            nn.Conv3d,
            in_channels,
            out_channels,
            groups,
            init_range,
            device,
            dtype,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPPaRAConv3d":
        instance = SPPaRAConv3d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            init_range,
            module.weight.device,
            module.weight.dtype,
        )
        copy_linear_params_(module, instance)
        return instance


def SPPaRA(
    module: AdaptableModule,
    init_range: float = _DEFAULT_INIT_RANGE,
    inplace=False,
    replacements=[
        (nn.Conv1d, SPPaRAConv1d),
        (nn.Conv2d, SPPaRAConv2d),
        (nn.Conv3d, SPPaRAConv3d),
    ],
):
    """Convert all submodules into {S}tructured {P}runing {Pa}rallel {Re}sidual {A}dapters"""
    mod = module
    for FromCls, ToSPPaRACls in replacements:
        mod = recursive_replace(
            mod,
            FromCls,
            ToSPPaRACls.from_module,
            inplace,
            None,
            init_range=init_range,
        )
    return mod


def named_parameters(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
    adapter_weights_only=True,
    in_features_mask: torch.BoolTensor = None,
    out_features_mask: torch.BoolTensor = None,
) -> Iterator[Tuple[str, nn.Parameter]]:
    for name, param in module.named_parameters(
        prefix=prefix, recurse=recurse, remove_duplicate=True
    ):
        if name == "adapter":
            if in_features_mask is not None:
                param = param[:, in_features_mask]
            if out_features_mask is not None:
                param = param[out_features_mask]
        elif name == "bias" and out_features_mask is not None:
            param = param[out_features_mask]
        elif name == "weight" and not adapter_weights_only:
            if out_features_mask is not None:
                param = param[out_features_mask]
            if in_features_mask is not None:
                param = param[:, in_features_mask]

        if adapter_weights_only:
            if name in {"adapter", "bias"}:
                yield (name, param)
        else:
            yield (name, param)


def parameters(
    module: nn.Module,
    recurse: bool = True,
    adapter_weights_only=True,
    in_features_mask: torch.BoolTensor = None,
    out_features_mask: torch.BoolTensor = None,
):
    for _, param in named_parameters(
        module, "", recurse, adapter_weights_only, in_features_mask, out_features_mask
    ):
        yield param
