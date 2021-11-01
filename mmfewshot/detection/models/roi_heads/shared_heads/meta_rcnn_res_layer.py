# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.roi_heads import ResLayer
from torch import Tensor


@SHARED_HEADS.register_module()
class MetaRCNNResLayer(ResLayer):
    """Shared resLayer for metarcnn and fsdetview.

    It provides different forward logics for query and support images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        out = out.mean(3).mean(2)
        return out

    def forward_support(self, x: Tensor) -> Tensor:
        """Forward function for support images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        x = self.max_pool(x)
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        out = self.sigmoid(out)
        out = out.mean(3).mean(2)
        return out
