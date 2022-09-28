from typing import Iterator, Tuple

import torch
from torch import nn

from .base import AdaptableModule
from .lora import LowRankMatrix
from .utils import bkron, copy_linear_params_, recursive_replace


def SPLoPA(
    module: AdaptableModule,
    num_prototypes: int = 64,
    block_shape: Tuple[int, int] = (32, 32),
    prototype_rank: int = 1,
    inplace=False,
):
    return recursive_replace(
        module,
        nn.Linear,
        SPLoPALinear.from_module,
        inplace,
        None,
        num_prototypes=num_prototypes,
        block_shape=block_shape,
        prototype_rank=prototype_rank,
    )


class SPLoPALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_prototypes: int = 64,
        block_shape: Tuple[int, int] = (32, 32),
        prototype_rank: int = 1,
        shared_prototypes: bool = True,
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

        self.adapter = SPLoPAdapter(
            (out_features, in_features),
            num_prototypes,
            block_shape,
            prototype_rank,
            shared_prototypes,
        )
        if bias:
            self.adapter_bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
            nn.init.uniform_(self.adapter_bias, -1e-4, 1e-4)
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
        result = None
        if self.adapter_bias is not None:
            result = self.bias + self.adapter_bias
        return result

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
        prototype_rank: int = 1,
    ) -> "SPLoPALinear":
        instance = SPLoPALinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            num_prototypes,
            block_shape,
            prototype_rank,
        )
        copy_linear_params_(module, instance)
        return instance

    def to_module(self) -> nn.Linear:
        instance = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        instance.weight = torch.nn.Parameter(self.adapted_weight)
        if self.bias is not None:
            instance.bias = torch.nn.Parameter(self.adapted_bias)
        return instance


class SPLoPAdapter(nn.Module):  # Inherit __setattr__
    def __init__(
        self,
        weight_shape: Tuple[int, int],
        num_prototypes: int = 64,
        block_shape: Tuple[int, int] = (32, 32),
        prototype_rank: int = 1,
        shared=True,
    ):
        nn.Module.__init__(self)

        n, m = weight_shape
        p, q = block_shape
        assert (
            n % p == 0 and m % q == 0
        ), f"Weight shape should be devisible by block shape, but found {weight_shape} and {block_shape}"

        Prototypes = shared_prototypes if shared else LowRankMatrix
        self.prototypes = Prototypes(num_prototypes, p, q, prototype_rank)
        self.pos_weights = nn.Parameter(torch.Tensor(num_prototypes, n // p, m // q))
        nn.init.uniform_(self.pos_weights, -1e-4, 1e-4)

    def __call__(self, weights: torch.Tensor):
        assert not weights.requires_grad
        return weights + torch.sum(bkron(self.pos_weights, self.prototypes()), dim=0)


def _shared_prototypes_singleton():
    _prototypes = {}

    def get_shared_prototype(n: int, p: int, q: int, rank: int = 1):
        nonlocal _prototypes
        key = (n, p, q, rank)
        if key not in _prototypes:
            _prototypes[key] = LowRankMatrix(n, p, q, rank)

        return _prototypes[key]

    return get_shared_prototype


shared_prototypes = _shared_prototypes_singleton()
