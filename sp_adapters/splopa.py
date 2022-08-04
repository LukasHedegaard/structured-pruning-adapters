import math
from typing import Iterator, Tuple

import torch
from torch import nn
from .utils import bkron
from .base import AdaptableModule


def SPLoPA(
    module: AdaptableModule,
    num_prototypes: int = 64,
    block_shape: Tuple[int, int] = (32, 32),
):
    return {nn.Linear: SPLoPALinear,}[
        type(module)
    ].from_module(module, num_prototypes, block_shape)


class SPLoPALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_prototypes: int = 64,
        block_shape: Tuple[int, int] = (32, 32),
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

        self.adapter = SPLoPAdapter(
            (out_features, in_features), num_prototypes, block_shape
        )
        if bias:
            self.adapter_bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
            nn.init.uniform_(self.adapter_bias, -1e-6, 1e-6)
        else:
            self.register_parameter("adapter_bias", None)
        self.reset_parameters()
        self.configure_parameter_read()

    def forward(self, input):
        return nn.functional.linear(input, self.adapted_weight, self.adapted_bias)

    @property
    def adapted_weight(self) -> nn.Parameter:
        return self.adapter(self.weight)

    @property
    def adapted_bias(self) -> nn.Parameter:
        if self.adapter_bias is not None:
            return self.bias + self.adapter_bias
        else:
            return self.adapter_bias

    def configure_parameter_read(
        self, adapter_weights_only=True, mask: torch.BoolTensor = None
    ):
        self._read_adapter_weights_only = adapter_weights_only
        self._read_mask = mask

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in super().named_parameters(prefix, recurse):
            if not self._read_adapter_weights_only or "adapter" in name:
                if name == "adapter.pos_weights" and self._read_mask is not None:
                    param = param[self._read_mask].flatten()
                yield (name, param)

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        num_prototypes: int = 64,
        block_shape: Tuple[int, int] = (32, 32),
    ) -> "SPLoPALinear":
        instance = SPLoPALinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            num_prototypes,
            block_shape,
        )
        copy_linear_params_(module, instance)
        return instance

    def to_module(self) -> nn.Linear:
        instance = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        instance.weight = torch.clone(self.adapt(self.weight))
        if self.bias is not None:
            instance.bias = self.bias
        return instance


def copy_linear_params_(source: nn.Linear, target: nn.Linear, clone=True):
    maybe_clone = torch.clone if clone else lambda x: x
    target.weight = nn.Parameter(maybe_clone(source.weight), requires_grad=False)
    if source.bias is not None:
        target.bias = nn.Parameter(maybe_clone(source.bias))


class SPLoPAdapter(nn.Module):  # Inherit __setattr__
    def __init__(
        self,
        weight_shape: Tuple[int, int],
        num_prototypes: int = 64,
        block_shape: Tuple[int, int] = (32, 32),
    ):
        nn.Module.__init__(self)

        n, m = weight_shape
        p, q = block_shape
        assert (
            n % p == 0 and m % q == 0
        ), "Weight shape should be devisible by block shape, but found {weight_shape} and {block_shape}"

        self.prototypes = shared_prototypes(num_prototypes, p, q)
        self.pos_weights = nn.Parameter(torch.Tensor(num_prototypes, n // p, m // q))
        nn.init.uniform_(self.pos_weights, -1e-6, 1e-6)

    def __call__(self, weights: torch.Tensor):
        assert not weights.requires_grad
        return weights + torch.sum(bkron(self.pos_weights, self.prototypes()), dim=0)

    def parameters(
        self, recurse: bool = True, mask: torch.Tensor = None
    ) -> Iterator[nn.Parameter]:
        it = super().parameters(recurse)
        return it


class LowRankMatrix(nn.Module):  # Inherit __setattr__
    def __init__(self, n: int, p: int, q: int):
        nn.Module.__init__(self)
        self.n, self.p, self.q = n, p, q
        self.cols = nn.Parameter(torch.Tensor(n, p, 1))
        self.rows = nn.Parameter(torch.Tensor(n, 1, q))
        self.reset_parameters()

    def __call__(self):
        return self.cols @ self.rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n}, {self.p}, {self.q})"

    def reset_parameters(self) -> None:
        # Init as in torch.nn.Linear.reset_parameters
        nn.init.kaiming_uniform_(self.cols, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rows, a=math.sqrt(5))


def _shared_prototypes_singleton():
    _prototypes = {}

    def get_shared_prototype(n: int, p: int, q: int):
        nonlocal _prototypes
        key = (n, p, q)
        if key not in _prototypes:
            _prototypes[key] = LowRankMatrix(n, p, q)

        return _prototypes[key]

    return get_shared_prototype


shared_prototypes = _shared_prototypes_singleton()
