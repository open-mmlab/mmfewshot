# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch.nn as nn
from mmcls.models.builder import BACKBONES
from torch import Tensor


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 is_pooling: bool = True,
                 padding: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if is_pooling:
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out


@BACKBONES.register_module()
class ConvNet(nn.Module):
    """Simple ConvNet.

    Args:
        depth (int): The number of `ConvBlock`.
        pooling_blocks (Sequence[int]): Indicate which block to use
            2x2 max pooling.
        padding_blocks (Sequence[int]): Indicate which block to use
            conv layer with padding.
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
    """

    def __init__(self,
                 depth: int,
                 pooling_blocks: Sequence[int],
                 padding_blocks: Sequence[int],
                 flatten: bool = True) -> None:
        super().__init__()
        layers = []
        for i in range(depth):
            in_channels = 3 if i == 0 else 64
            out_channels = 64
            # only pooling for fist 4 layers
            layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    is_pooling=(i in pooling_blocks),
                    padding=1 if i in padding_blocks else 0))
        self.flatten = flatten
        self.layers = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)  # (N, 64, 5, 5)
        if self.flatten:
            out = out.view(out.size(0), -1)  # (N, 1600)
        return out

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@BACKBONES.register_module()
class Conv4(ConvNet):

    def __init__(self,
                 depth: int = 4,
                 pooling_blocks: Sequence[int] = (0, 1, 2, 3),
                 padding_blocks: Sequence[int] = (0, 1, 2, 3),
                 flatten: bool = True) -> None:
        super().__init__(
            depth=depth,
            pooling_blocks=pooling_blocks,
            padding_blocks=padding_blocks,
            flatten=flatten)


@BACKBONES.register_module()
class Conv4NoPool(ConvNet):
    """Used for RelationNet."""

    def __init__(self,
                 depth: int = 4,
                 pooling_blocks: Sequence[int] = (0, 1),
                 padding_blocks: Sequence[int] = (2, 3),
                 flatten: bool = False) -> None:
        super().__init__(
            depth=depth,
            pooling_blocks=pooling_blocks,
            padding_blocks=padding_blocks,
            flatten=flatten)
