from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import ConvFCBBoxHead
from torch import Tensor


@HEADS.register_module()
class TwoBranchBBoxHead(ConvFCBBoxHead):
    """BBox head for `MPSR <https://arxiv.org/abs/2007.09384>`_.

    Args:
        auxiliary_loss_weight (float): Weight of auxiliary loss Default: 0.1.
    """

    def __init__(self,
                 auxiliary_loss_weight: float = 0.1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.auxiliary_avg_pooling = nn.AdaptiveAvgPool2d(self.roi_feat_size)
        assert auxiliary_loss_weight >= 0
        self.auxiliary_loss_weight = auxiliary_loss_weight

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function for main data."""
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for i, fc in enumerate(self.cls_fcs):
            if (i + 1) == len(self.cls_fcs):
                x_cls = fc(x_cls)
            else:
                x_cls = self.relu(fc(x_cls))
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def forward_auxiliary_single(self, x: Tensor) -> Tuple[Tensor, ]:
        """Forward function for auxiliary of single scale."""
        x = self.auxiliary_avg_pooling(x)
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for i, fc in enumerate(self.cls_fcs):
            if (i + 1) == len(self.cls_fcs):
                x_cls = fc(x_cls)
            else:
                x_cls = self.relu(fc(x_cls))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        return cls_score,

    def forward_auxiliary(self, x: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward auxiliary features at multiple scales.

        Args:
            x (tuple[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.

        Returns:
            tuple[Tensor]: Classification scores for all scale levels, each is
                a 4D-tensor, the channels number is num_anchors * num_classes.
        """
        return multi_apply(self.forward_auxiliary_single, x)

    @force_fp32(apply_to=('cls_score'))
    def auxiliary_loss(self,
                       cls_score: Tensor,
                       labels: Tensor,
                       label_weights: Tensor,
                       reduction_override: Optional[str] = None) -> Dict:
        """Compute loss for auxiliary features.

        Args:
            cls_score (Tensor): Classification scores for all scales with
                shape (num_proposals, num_classes).
            labels (Tensor): Labels of each proposal with shape
                (num_proposals).
            label_weights (Tensor): Label weights of each proposal with shape
                (num_proposals).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            loss_cls_ = self.auxiliary_loss_weight * self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['loss_cls_auxiliary'] = loss_cls_
            losses['acc_auxiliary'] = accuracy(cls_score, labels)
        return losses
