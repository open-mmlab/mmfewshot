# This code is modified from https://github.com/nupurkmr9/S2M2_fewshot

from typing import Tuple, Union

import torch.nn as nn
from mmcls.models.builder import BACKBONES
from torch import Tensor


class WRNBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 drop_rate: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(p=drop_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=True))

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        if self.drop_rate > 0.0:
            out = self.dropout(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


@BACKBONES.register_module()
class WideResNet(nn.Module):
    """WideResNet.

    Args:
        depth (int): The number of layers.
        widen_factor (int): The widen factor of channels. Default: 1.
        stride (int): Stride of first layer. Default: 1.
        drop_rate (float): Dropout rate. Default: 0.0.
        with_avgpool (bool): Whether to average pool the features.
            Default: True.
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
        pool_size (tuple(int,int)): The output shape of average pooling layer.
            Default: (1, 1).
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int = 1,
        stride: int = 1,
        drop_rate: float = 0.0,
        flatten: bool = True,
        with_avgpool: bool = True,
        pool_size: Tuple[int, int] = (1, 1)
    ) -> None:  # noqa: E125

        super().__init__()
        num_channels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        num_layers = (depth - 4) / 6
        block = WRNBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st layer
        self.layer1 = self._make_layer(num_layers, num_channels[0],
                                       num_channels[1], block, stride,
                                       drop_rate)
        # 2nd layer
        self.layer2 = self._make_layer(num_layers, num_channels[1],
                                       num_channels[2], block, 2, drop_rate)
        # 3rd layer
        self.layer3 = self._make_layer(num_layers, num_channels[2],
                                       num_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.norm1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels[3]

        self.flatten = flatten
        self.with_avgpool = with_avgpool
        if self.with_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

    @staticmethod
    def _make_layer(num_layers: Union[int, float], in_channels: int,
                    out_channels: int, block: nn.Module, stride: int,
                    drop_rate: float) -> nn.Sequential:
        layers = []
        for i in range(int(num_layers)):
            layers.append(
                block(i == 0 and in_channels or out_channels, out_channels,
                      i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.with_avgpool:
            x = self.avgpool(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return x

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@BACKBONES.register_module()
class WRN28x10(WideResNet):

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        stride: int = 1,
        drop_rate: float = 0.5,
        flatten: bool = True,
        with_avgpool: bool = True,
        pool_size: Tuple[int, int] = (1, 1)
    ) -> None:  # noqa: E125

        super().__init__(
            depth=depth,
            widen_factor=widen_factor,
            stride=stride,
            drop_rate=drop_rate,
            flatten=flatten,
            with_avgpool=with_avgpool,
            pool_size=pool_size)
