from typing import Union

from torch import nn

AdaptableModule = Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]
