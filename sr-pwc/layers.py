import torch
import torch.nn as nn

import numpy as np

from network_utils import init_weights

class SubPixConv(nn.Module):

    """ Efficient Sub-Pixel Convolutional Layer (Shi et al. 2016) """

    def __init__(self, in_channels, out_channels=None, upscale_factor=2, k=3, s=1, p=1):
        super(SubPixConv, self).__init__()

        if out_channels is None:
            out_channels = in_channels * (upscale_factor**2)

        self.conv = nn.Conv2d(in_channels, out_channels, 
                kernel_size=k, stride=s, padding=p)
        self.shuf = nn.PixelShuffle(upscale_factor)

        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuf(x)
        return x


class ResBlock2(nn.Module):

    """ Residual Block without Bottleneck """

    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super(ResBlock2, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=s, padding=p))

        self.apply(init_weights)

    def forward(self, x):
        identity = x
        r = self.block(x)
        return identity + r

