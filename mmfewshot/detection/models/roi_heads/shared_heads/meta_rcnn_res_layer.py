import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.roi_heads import ResLayer
from torch import Tensor


@SHARED_HEADS.register_module()
class MetaRCNNResLayer(ResLayer):
    """Shared resLayer for metarcnn and fsdetview."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    @auto_fp16()
    def forward(self, x: Tensor, is_support: bool = False) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).
            is_support (bool): Whether `x` is support features or not.
                Default: False.

        Returns:
            Tensor: Shape of (N, C).
        """
        if is_support:
            x = self.max_pool(x)
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        if is_support:
            out = self.sigmoid(out)
        out = out.mean(3).mean(2)
        return out
