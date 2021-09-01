import copy

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BBoxHead


@HEADS.register_module()
class MetaBBoxHead(BBoxHead):
    """BBoxHead with meta classification.

    Args:
        num_meta_classes (int): Number of classes for meta classification.
        meta_cls_in_channels (int): Number of input feature channels.
        with_meta_cls_loss (bool): Use meta classification loss.
            Default: True.
        loss_meta (dict): Config for meta classification loss.
    """

    def __init__(self,
                 num_meta_classes,
                 meta_cls_in_channels=2048,
                 with_meta_cls_loss=True,
                 loss_meta=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super(MetaBBoxHead, self).__init__(*args, **kwargs)
        self.with_meta_cls_loss = with_meta_cls_loss
        if with_meta_cls_loss:
            self.fc_meta = nn.Linear(meta_cls_in_channels, num_meta_classes)
            self.loss_meta_cls = build_loss(copy.deepcopy(loss_meta))

    def forward_meta_cls(self, support_feat):
        """Forward function for meta classification.

        Args:
            support_feat (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        meta_cls_score = self.fc_meta(support_feat)
        return meta_cls_score

    @force_fp32(apply_to='meta_cls_score')
    def loss_meta(self,
                  meta_cls_score,
                  meta_cls_labels,
                  meta_cls_label_weights,
                  reduction_override=None):
        """Meta classification loss.

        Args:
            meta_cls_score (Tensor): Predicted meta classification scores
                 with shape (N, num_meta_classes).
            meta_cls_labels (Tensor): Corresponding class indices with
                shape (N).
            meta_cls_label_weights (Tensor): Meta classification loss weight
                of each sample with shape (N).
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        losses = dict()
        avg_factor = max(
            torch.sum(meta_cls_label_weights > 0).float().item(), 1.)
        if meta_cls_score.numel() > 0:
            loss_meta_cls_ = self.loss_meta_cls(
                meta_cls_score,
                meta_cls_labels,
                meta_cls_label_weights,
                reduction_override=reduction_override)
            if isinstance(loss_meta_cls_, dict):
                losses.update(loss_meta_cls_ / avg_factor)
            else:
                losses['loss_meta_cls'] = loss_meta_cls_ / avg_factor
            losses['meta_acc'] = accuracy(meta_cls_score, meta_cls_labels)
        return losses
