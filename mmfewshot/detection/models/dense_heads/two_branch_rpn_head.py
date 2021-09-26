import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class TwoBranchRPNHead(RPNHead):
    """RPN head for `MPSR <https://arxiv.org/abs/2007.09384>`_.

    Args:
        mid_channels (int): Input channels of `rpn_cls_conv`. Default: 64.
    """

    def __init__(self, mid_channels=64, **kwargs):
        self.mid_channels = mid_channels
        super(TwoBranchRPNHead, self).__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls_conv = nn.Conv2d(self.feat_channels,
                                      self.num_anchors * self.mid_channels, 1)
        self.rpn_cls = nn.Conv2d(self.mid_channels, self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def forward_single(self, feat):
        """Forward feature map of a single scale level."""
        feat = self.rpn_conv(feat)
        feat = F.relu(feat, inplace=True)
        cls_feat = self.rpn_cls_conv(feat)
        b, _, h, w = cls_feat.size()
        cls_feat = cls_feat.view(b * self.num_anchors, self.mid_channels, h, w)
        rpn_cls_score = self.rpn_cls(cls_feat).view(-1, self.num_anchors, h, w)
        rpn_bbox_pred = self.rpn_reg(feat)
        return rpn_cls_score, rpn_bbox_pred

    def forward_auxiliary_single(self, feat):
        """Forward auxiliary feature map of a single scale level."""
        feat = self.rpn_conv(feat)
        feat = F.relu(feat, inplace=True)
        # only use center 2x2(last 3x3)
        feat = feat[:, :, 3:-3, 3:-3]
        cls_feat = self.rpn_cls_conv(feat)
        b, _, h, w = cls_feat.size()
        cls_feat = cls_feat.view(b * self.num_anchors, self.mid_channels, h, w)
        rpn_cls_score = self.rpn_cls(cls_feat).view(-1, self.num_anchors, h, w)
        return rpn_cls_score,

    def forward_auxiliary(self, feats):
        """Forward auxiliary features at multiple scales.

        Args:
            feats (tuple[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.

        Returns:
            list[Tensor]: Classification scores for all scale levels, each is
                a 4D-tensor, the channels number is num_anchors * num_classes.
        """
        return multi_apply(self.forward_auxiliary_single, feats)

    def forward_auxiliary_train(self, feats):
        """Forward function and calculate loss for auxiliary data in training.

        Args:
            feats (tuple[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        outs, = self.forward_auxiliary(feats)
        losses = self.auxiliary_loss(outs)
        return losses

    @force_fp32(apply_to=('cls_scores'))
    def auxiliary_loss(self, cls_scores):
        """Compute losses for auxiliary features.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        device = cls_scores[0].device
        labels_list = [
            torch.zeros_like(cls_score, dtype=torch.long).to(device)
            for cls_score in cls_scores
        ]
        label_weights_list = [
            torch.ones_like(cls_score, dtype=torch.float).to(device)
            for cls_score in cls_scores
        ]
        num_total_samples = sum(label_weights.sum()
                                for label_weights in label_weights_list)
        losses, = multi_apply(
            self.auxiliary_loss_single,
            cls_scores,
            labels_list,
            label_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls_auxiliary=losses)

    def auxiliary_loss_single(self, cls_score, labels, label_weights,
                              num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            num_total_samples (int): The number of positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss only
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        losses_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        return losses_cls,
