import os
import torch
import torch.nn as nn
from torch import autograd, optim

import torch.nn.functional as F
import random
import torchvision.datasets as dataset
import torchvision.datasets




# Generator residual Block
class GResidualBlock(nn.Module):
    '''
    GResidualBlock Class
    Values:
    c_dim: the dimension of conditional vector [c, z], a scalar
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    '''

    def __init__(self, c_dim, in_channels, out_channels):
        super().__init__()

        self.conv1 =  DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv2 =  DepthwiseSeparableConv(out_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = ClassConditionalBatchNorm2d(c_dim, in_channels)
        self.bn2 = ClassConditionalBatchNorm2d(c_dim, out_channels)

        self.activation = nn.ReLU()
        self.upsample_fn = nn.Upsample(scale_factor=2)     # upsample occurs in every gblock

        self.mixin = (in_channels != out_channels)
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, x, y):
        # h := upsample(x, y)
        h = self.bn1(x, y)
        h = self.activation(h)
        h = self.upsample_fn(h)
        h = self.conv1(h)

        # h := conv(h, y)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)

        # x := upsample(x)
        x = self.upsample_fn(x)
        if self.mixin:
            x = self.conv_mixin(x)

        return h + x



# Discriminator residual Block
class DResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=True, use_preactivation=False):
        super().__init__()

        self.conv1 =  DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 =  DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)


        self.activation = nn.ReLU()
        self.use_preactivation = use_preactivation  # apply preactivation in all except first dblock

        self.downsample = downsample    # downsample occurs in all except last dblock

        if downsample:
            self.downsample_fn = nn.AvgPool2d(2)
        self.mixin = (in_channels != out_channels) or downsample

        if self.mixin:
            self.conv_mixin =  DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, padding=0)

    def _residual(self, x):
        if self.use_preactivation:
            if self.mixin:
                x = self.conv_mixin(x)
            if self.downsample:
                x = self.downsample_fn(x)
        else:
            if self.downsample:
                x = self.downsample_fn(x)
            if self.mixin:
                x = self.conv_mixin(x)
        return x

    def forward(self, x):
        # Apply preactivation if applicable
        if self.use_preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.bn1(h)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.bn2(h)
        h = self.activation(h)
        h = self.conv2(h)

        if self.downsample:
            h = self.downsample_fn(h)

        return h + self._residual(x)
