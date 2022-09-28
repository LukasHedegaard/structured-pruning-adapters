import math

import torch
from torch import nn


class LowRankMatrix(nn.Module):  # Inherit __setattr__
    def __init__(self, n: int, p: int, q: int, rank: int = 1, init_near_zero=False):
        nn.Module.__init__(self)
        self.n, self.p, self.q = n, p, q
        self.cols = nn.Parameter(torch.Tensor(n, p, rank))
        self.rows = nn.Parameter(torch.Tensor(n, rank, q))
        self.reset_parameters(init_near_zero)

    def __call__(self):
        return self.cols @ self.rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n}, {self.p}, {self.q})"

    def reset_parameters(self, init_near_zero=False) -> None:
        # Init as in torch.nn.Linear.reset_parameters
        nn.init.kaiming_uniform_(self.cols, a=math.sqrt(5))
        if init_near_zero:
            nn.init.uniform_(self.rows, -1e-4, 1e-4)
        else:
            nn.init.kaiming_uniform_(self.rows, a=math.sqrt(5))
