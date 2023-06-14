from logging import getLogger
from typing import Iterator, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from .base import AdaptableModule
from .lora import LowRankMatrix
from .utils import copy_linear_params_, recursive_replace

logger = getLogger(__name__)

_DEFAULT_RANK = 16
_DEFAULT_INIT_RANGE = 1e-4


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
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.adapter = LowRankMatrix(
            1, in_features, out_features, rank, init_range=init_range
        )
        self.reset_parameters()

    def forward(self, input):
        return nn.functional.linear(input, self.adapted_weight, self.bias)

    @property
    def adapted_weight(self) -> nn.Parameter:
        if self.weight.requires_grad:
            self.weight.requires_grad = False
            logger.warning("Forcing `weight.requires_grad = False`")
        return self.adapter().squeeze(0) + self.weight

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
            instance.bias = torch.nn.Parameter(self.bias)
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

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.adapted_weight, self.bias)

    @property
    def adapted_weight(self) -> nn.Parameter:
        if self.weight.requires_grad:
            self.weight.requires_grad = False
        w_diag = torch.zeros_like(self.weight)
        kdx = self.kernel_size[0] // 2
        center_inds = [kdx for _ in range(self._nd)]
        # w_diag[:, :, *center_inds] += self.adapter().squeeze(0) # Python 3.11+
        wdx = [slice(None), slice(None)] + center_inds
        w_diag.__setitem__(wdx, w_diag.__getitem__(wdx) + self.adapter().squeeze(0))
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
        assert self.kernel_size[0] % 2 == 1, "kernel_size should be odd."
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
        assert self.kernel_size[0] % 2 == 1, "kernel_size should be odd."
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
        assert self.kernel_size[0] % 2 == 1, "kernel_size should be odd."
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


class SPLoRAMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.Module.__init__(self)
        self.rank = rank
        self.init_range = init_range
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # Force module to handle embedding dims separately
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = SPLoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False,
            rank=rank,
            init_range=init_range,
            **factory_kwargs,
        )
        self.k_proj = SPLoRALinear(
            in_features=self.kdim,
            out_features=embed_dim,
            bias=False,
            rank=rank,
            init_range=init_range,
            **factory_kwargs,
        )
        self.v_proj = SPLoRALinear(
            in_features=self.vdim,
            out_features=embed_dim,
            bias=False,
            rank=rank,
            init_range=init_range,
            **factory_kwargs,
        )
        # self.register_parameter("in_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = SPLoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
            rank=rank,
            init_range=init_range,
            **factory_kwargs,
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    @property
    def in_proj_weight(self) -> nn.Parameter:
        return (
            torch.cat(
                (
                    self.q_proj.adapted_weight,
                    self.k_proj.adapted_weight,
                    self.v_proj.adapted_weight,
                )
            )
            if self._qkv_same_embed_dim
            else None
        )

    def forward(  # noqa: C901
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            is_causal: If specified, applies a causal mask as attention mask.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``attn_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.adapted_weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                [
                    (x is None or x.is_cuda or "cpu" in str(x.device))
                    for x in tensor_args
                ]
            ):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(
                [x is not None and x.requires_grad for x in tensor_args]
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.adapted_weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type,
                )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # if not self._qkv_same_embed_dim:
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.adapted_weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.adapted_weight,
                k_proj_weight=self.k_proj.adapted_weight,
                v_proj_weight=self.v_proj.adapted_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.adapted_weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    @classmethod
    def from_module(
        cls,
        module: nn.MultiheadAttention,
        # rank (int) or fraction of output_channels
        rank: Union[int, float] = _DEFAULT_RANK,
        init_range: float = _DEFAULT_INIT_RANGE,
    ) -> "SPLoRAMultiheadAttention":
        instance = SPLoRAMultiheadAttention(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
            batch_first=module.batch_first,
            rank=rank,
            init_range=init_range,
        )
        if module.in_proj_weight is not None:
            (instance.q_proj.weight, instance.k_proj.weight, instance.v_proj.weight) = [
                nn.Parameter(w, requires_grad=False)
                for w in torch.clone(module.in_proj_weight).chunk(3)
            ]
        else:
            instance.q_proj.weight = nn.Parameter(
                torch.clone(module.q_proj_weight), requires_grad=False
            )
            instance.k_proj.weight = nn.Parameter(
                torch.clone(module.k_proj_weight), requires_grad=False
            )
            instance.v_proj.weight = nn.Parameter(
                torch.clone(module.v_proj_weight), requires_grad=False
            )

        instance.out_proj.weight = nn.Parameter(
            torch.clone(module.out_proj.weight), requires_grad=False
        )

        if module.in_proj_bias is not None:
            instance.in_proj_bias = nn.Parameter(torch.clone(module.in_proj_bias))
        if module.out_proj.bias is not None:
            instance.out_proj.bias = nn.Parameter(torch.clone(module.out_proj.bias))
        if module.bias_k is not None:
            instance.bias_k = nn.Parameter(torch.clone(module.bias_k))
        if module.bias_v is not None:
            instance.bias_v = nn.Parameter(torch.clone(module.bias_v))

        return instance

    def to_module(self) -> nn.MultiheadAttention:
        instance = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=self.in_proj_bias is not None,
            add_bias_kv=self.bias_k is not None,
            add_zero_attn=self.add_zero_attn,
            kdim=self.kdim,
            vdim=self.vdim,
            batch_first=self.batch_first,
        )

        if self._qkv_same_embed_dim:
            instance.in_proj_weight = nn.Parameter(torch.clone(self.in_proj_weight))
        else:
            instance.q_proj_weight = nn.Parameter(
                torch.clone(self.q_proj.adapted_weight)
            )
            instance.k_proj_weight = nn.Parameter(
                torch.clone(self.k_proj.adapted_weight)
            )
            instance.v_proj_weight = nn.Parameter(
                torch.clone(self.v_proj.adapted_weight)
            )

        instance.out_proj.weight = nn.Parameter(
            torch.clone(self.out_proj.adapted_weight)
        )

        if self.in_proj_bias is not None:
            instance.in_proj_bias = nn.Parameter(torch.clone(self.in_proj_bias))
        if self.out_proj.bias is not None:
            instance.out_proj.bias = nn.Parameter(torch.clone(self.out_proj.bias))
        if self.bias_k is not None:
            instance.bias_k = nn.Parameter(torch.clone(self.bias_k))
        if self.bias_v is not None:
            instance.bias_v = nn.Parameter(torch.clone(self.bias_v))

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
        (nn.MultiheadAttention, SPLoRAMultiheadAttention),
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
        if name == "adapter.rows" and in_features_mask is not None:
            param = param[:, :, in_features_mask]
        elif name == "adapter.cols" and out_features_mask is not None:
            param = param[:, out_features_mask, :]
        elif name == "bias" and out_features_mask is not None:
            param = param[out_features_mask]
        elif name == "weight" and not adapter_weights_only:
            if out_features_mask is not None:
                param = param[out_features_mask]
            if in_features_mask is not None:
                param = param[:, in_features_mask]

        if adapter_weights_only:
            if name in {"adapter.rows", "adapter.cols", "bias"}:
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
