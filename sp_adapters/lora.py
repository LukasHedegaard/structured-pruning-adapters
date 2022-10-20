from typing import Union
import math
import torch
from torch import nn


class LowRankMatrix(nn.Module):  # Inherit __setattr__
    def __init__(
        self,
        num_filters: int,
        in_features: int,
        out_features: int,
        rank: Union[int, float] = 1,  # rank (int) or fraction of output_channels
        init_range: float = None,
    ):
        nn.Module.__init__(self)
        if rank < 1:
            rank = round(out_features * rank)
        else:
            rank = round(rank)
        self.rank = rank
        self.num_filters = num_filters
        self.in_features = in_features
        self.out_features = out_features
        self.rows = nn.Parameter(torch.Tensor(num_filters, rank, in_features))
        self.cols = nn.Parameter(torch.Tensor(num_filters, out_features, rank))
        self.reset_parameters(init_range)

    def __call__(self):
        return self.cols @ self.rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_filters={self.num_filters}, in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"

    def reset_parameters(self, init_range=None) -> None:
        # Init as in torch.nn.Linear.reset_parameters
        nn.init.kaiming_uniform_(self.rows, a=math.sqrt(5))
        if init_range:
            nn.init.uniform_(self.cols, -init_range, init_range)
        else:
            nn.init.kaiming_uniform_(self.cols, a=math.sqrt(5))
