import math
from typing import Union

import torch
from torch import nn


class LowRankMatrix(nn.Module):  # Inherit __setattr__
    def __init__(
        self,
        num_filters: int,
        in_features: int,
        out_features: int,
        rank: Union[int, float] = 1,  # rank (int) or fraction of output_channels
        init_near_zero=False,
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
        self.cols = nn.Parameter(torch.Tensor(num_filters, out_features, rank))
        self.rows = nn.Parameter(torch.Tensor(num_filters, rank, in_features))
        self.reset_parameters(init_near_zero)

    def __call__(self):
        return self.cols @ self.rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_filters={self.num_filters}, in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"

    def reset_parameters(self, init_near_zero=False) -> None:
        # Init as in torch.nn.Linear.reset_parameters
        nn.init.kaiming_uniform_(self.cols, a=math.sqrt(5))
        if init_near_zero:
            nn.init.uniform_(self.rows, -1e-4, 1e-4)
        else:
            nn.init.kaiming_uniform_(self.rows, a=math.sqrt(5))
