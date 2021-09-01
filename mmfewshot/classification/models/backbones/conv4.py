import torch.nn as nn
from mmcls.models.builder import BACKBONES


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_pooling=True, padding=1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        layers = [self.conv, self.bn, self.relu]
        if is_pooling:
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


@BACKBONES.register_module()
class ConvNet(nn.Module):
    """Simple ConvNet.

    Args:
        depth (int): The number of `ConvBlock`.
        pooling_layers (list[int]): Indicate each block whether to use
            max pooling.
        padding_layers (list[int]): Indicate each block whether to pad
            in conv layer.
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
    """

    def __init__(self, depth, pooling_layers, padding_layers, flatten=True):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(depth):
            in_channels = 3 if i == 0 else 64
            out_channels = 64
            # only pooling for fist 4 layers
            layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    is_pooling=(i in pooling_layers),
                    padding=1 if i in padding_layers else 0))
        self.flatten = flatten
        self.layers = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        out = self.layers(x)  # (N, 64, 5, 5)
        if self.flatten:
            out = out.view(out.size(0), -1)  # (N, 1600)
        return out

    def init_weights(self):
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
                 depth=4,
                 pooling_layers=(0, 1, 2, 3),
                 padding_layers=(0, 1, 2, 3),
                 flatten=True):
        super(Conv4, self).__init__(
            depth=depth,
            pooling_layers=pooling_layers,
            padding_layers=padding_layers,
            flatten=flatten)


@BACKBONES.register_module()
class Conv4NoPool(ConvNet):
    """Used for RelationNet."""

    def __init__(self,
                 depth=4,
                 pooling_layers=(0, 1),
                 padding_layers=(2, 3),
                 flatten=False):
        super(Conv4NoPool, self).__init__(
            depth=depth,
            pooling_layers=pooling_layers,
            padding_layers=padding_layers,
            flatten=flatten)
