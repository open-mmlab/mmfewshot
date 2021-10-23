from typing import Tuple

from mmcv.cnn import build_conv_layer
from mmdet.models import ResNet
from mmdet.models.builder import BACKBONES
from torch import Tensor


@BACKBONES.register_module()
class ResNetWithMetaConv(ResNet):
    """ResNet with `meta_conv` to handle different inputs.

    When input with shape (N, 3, H, W) from images, the network will use
    `conv1` as regular ResNet. When input with shape (N, 4, H, W) from (image +
    mask) the network will replace `conv1` with `meta_conv` to handle
    additional channel.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.meta_conv = build_conv_layer(
            self.conv_cfg,
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)

    def forward(self, x: Tensor, use_meta_conv: bool = False) -> Tuple[Tensor]:
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (N, 3, H, W) from images
                or (N, 4, H, W) from (images + masks).
            use_meta_conv (bool): If set True, forward input tensor with
                `meta_conv` which require tensor with shape (N, 4, H, W).
                Otherwise, forward input tensor with `conv1` which require
                tensor with shape (N, 3, H, W). Default: False.

        Returns:
            tuple[Tensor]: Tuple of features, each item with
                shape (N, C, H, W).
        """
        if use_meta_conv:
            x = self.meta_conv(x)
        else:
            x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
