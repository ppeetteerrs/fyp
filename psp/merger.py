from torch import nn
from torch.nn import Conv2d

from psp.encoder import ResnetBlock


class Merger(nn.Sequential):
    def __init__(self, in_channel: int, mid_channel: int, n_conv: int):
        super().__init__(
            ResnetBlock(in_channel, mid_channel, 1),
            *[ResnetBlock(mid_channel, mid_channel, 1) for _ in range(n_conv - 1)],
            Conv2d(mid_channel, 1, (1, 1), 1, bias=True),
        )
