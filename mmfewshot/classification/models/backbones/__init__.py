from mmcls.models.builder import BACKBONES

from .conv4 import Conv4, ConvNet
from .resnet12 import ResNet12
from .wrn import WideResNet, WRN28x10

__all__ = [
    'BACKBONES', 'ResNet12', 'Conv4', 'ConvNet', 'WRN28x10', 'WideResNet'
]
