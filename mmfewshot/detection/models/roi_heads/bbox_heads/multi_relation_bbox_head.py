# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BBoxHead
from torch import Tensor


@HEADS.register_module()
class MultiRelationBBoxHead(BBoxHead):
    """BBox head for `Attention RPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        patch_relation (bool): Whether use patch_relation head for
            classification. Following the official implementation,
            `patch_relation` always be True, because only patch relation
            head contain regression head. Default: True.
        local_correlation (bool): Whether use local_correlation head for
            classification. Default: True.
        global_relation (bool): Whether use global_relation head for
            classification. Default: True.
    """

    def __init__(self,
                 patch_relation: bool = True,
                 local_correlation: bool = True,
                 global_relation: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # remove unused parameters inherited from BBoxHead
        if hasattr(self, 'fc_cls'):
            del self.fc_cls
        if hasattr(self, 'fc_reg'):
            del self.fc_reg

        # following the official implementation patch relation must be True,
        # because only patch relation head contain regression head
        self.patch_relation = True
        self.local_correlation = local_correlation
        self.global_relation = global_relation

        if self.patch_relation:
            self.patch_relation_branch = nn.Sequential(
                nn.Conv2d(
                    self.in_channels * 2,
                    int(self.in_channels / 4),
                    1,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True),
                # 7x7 -> 5x5
                nn.AvgPool2d(kernel_size=3, stride=1),
                # 5x5 -> 3x3
                nn.Conv2d(
                    int(self.in_channels / 4),
                    int(self.in_channels / 4),
                    3,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    int(self.in_channels / 4),
                    self.in_channels,
                    1,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True),
                # 3x3 -> 1x1
                nn.AvgPool2d(kernel_size=3, stride=1))
            self.patch_relation_fc_reg = nn.Linear(self.in_channels, 4)
            self.patch_relation_fc_cls = nn.Linear(self.in_channels, 2)

        if self.local_correlation:
            self.local_correlation_branch = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    1,
                    padding=0,
                    bias=False))
            self.local_correlation_fc_cls = nn.Linear(self.in_channels, 2)

        if self.global_relation:
            self.global_relation_avgpool = nn.AvgPool2d(7)
            self.global_relation_branch = nn.Sequential(
                nn.Linear(self.in_channels * 2, self.in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_channels, self.in_channels),
                nn.ReLU(inplace=True))
            self.global_relation_fc_cls = nn.Linear(self.in_channels, 2)

    def forward(self, query_feat: Tensor,
                support_feat: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            query_feat (Tensor): Shape of (num_proposals, C, H, W).
            support_feat (Tensor): Shape of (1, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """

        # global_relation
        if self.global_relation:
            global_query_feat = self.global_relation_avgpool(
                query_feat).squeeze(3).squeeze(2)
            global_support_feat = self.global_relation_avgpool(
                support_feat).squeeze(3).squeeze(2).expand_as(
                    global_query_feat)
            global_feat = \
                torch.cat((global_query_feat, global_support_feat), 1)
            global_feat = self.global_relation_branch(global_feat)
            global_relation_cls_score = \
                self.global_relation_fc_cls(global_feat)

        # local_correlation
        if self.local_correlation:
            local_query_feat = self.local_correlation_branch(query_feat)
            local_support_feat = self.local_correlation_branch(support_feat)
            local_feat = F.conv2d(
                local_query_feat,
                local_support_feat.permute(1, 0, 2, 3),
                groups=2048)
            local_feat = F.relu(local_feat, inplace=True).squeeze(3).squeeze(2)
            local_correlation_cls_score = self.local_correlation_fc_cls(
                local_feat)

        # patch_relation
        if self.patch_relation:
            patch_feat = torch.cat(
                (query_feat, support_feat.expand_as(query_feat)), 1)
            # 7x7 -> 1x1
            patch_feat = self.patch_relation_branch(patch_feat)
            patch_feat = patch_feat.squeeze(3).squeeze(2)
            patch_relation_cls_score = self.patch_relation_fc_cls(patch_feat)
            patch_relation_bbox_pred = self.patch_relation_fc_reg(patch_feat)

        # aggregate multi relation result
        # following the official implementation patch,
        # only patch relation head contain regression head
        bbox_pred_all = patch_relation_bbox_pred
        cls_score_all = patch_relation_cls_score
        if self.local_correlation:
            cls_score_all += local_correlation_cls_score
        if self.global_relation:
            cls_score_all += global_relation_cls_score
        return cls_score_all, bbox_pred_all

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        num_pos_pair_samples: int,
        reduction_override: Optional[str] = None,
        sample_fractions: Sequence[Union[int, float]] = (1, 2, 1)
    ) -> Dict:
        """Compute losses of the head.

        Args:
            cls_scores (Tensor): Box scores with shape of
                (num_proposals, num_classes)
            bbox_preds (Tensor): Box energies / deltas with shape
                of (num_proposals, num_classes * 4)
            rois (Tensor): shape (N, 4) or (N, 5)
            labels (Tensor): Labels of proposals with shape (num_proposals).
            label_weights (Tensor): Label weights of proposals with shape
                (num_proposals).
            bbox_targets (Tensor): BBox regression targets of each proposal
                weight with shape (num_proposals, num_classes * 4).
            bbox_weights (Tensor): BBox regression loss weights of each
                proposal with shape (num_proposals, num_classes * 4).
            num_pos_pair_samples (int): Number of samples from positive pairs.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.
            sample_fractions (Sequence[int | float]):
                Fractions of positive samples, negative samples from positive
                pair, negative samples from negative pair. Default: (1, 2, 1).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        # fg bg sampling
        num_instances = labels.size(0)
        fg_samples_inds = torch.nonzero(
            labels == 0, as_tuple=False).squeeze(-1)
        bg_samples_inds = torch.nonzero(
            labels == 1, as_tuple=False).squeeze(-1)

        bg_cls_scores = cls_scores[bg_samples_inds, :]

        num_pos_pair_bg_samples = max(
            1,
            min(fg_samples_inds.shape[0] * sample_fractions[1],
                int(num_instances / sum(sample_fractions))))
        num_neg_pair_samples = max(
            1,
            min(fg_samples_inds.shape[0] * sample_fractions[2],
                num_pos_pair_bg_samples))

        _, sorted_inds = torch.sort(bg_cls_scores[:, 0], descending=True)
        sorted_bg_samples_inds = bg_samples_inds[sorted_inds]
        pos_pair_bg_samples_inds = sorted_bg_samples_inds[
            sorted_bg_samples_inds <
            num_pos_pair_samples][:num_pos_pair_bg_samples]
        neg_pair_samples_inds = sorted_bg_samples_inds[
            sorted_bg_samples_inds >=
            num_pos_pair_samples][:num_neg_pair_samples]

        topk_inds = torch.cat(
            [fg_samples_inds, pos_pair_bg_samples_inds, neg_pair_samples_inds],
            dim=0)

        if cls_scores is not None:
            if cls_scores.numel() > 0:
                # cls_inds resample the rois to get final classification loss
                losses['loss_cls'] = self.loss_cls(
                    cls_scores[topk_inds],
                    labels[topk_inds],
                    label_weights[topk_inds],
                    avg_factor=len(topk_inds),
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_scores, labels)
        if bbox_preds is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_preds = self.bbox_coder.decode(
                        rois[:, 1:], bbox_preds)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_preds.view(
                        bbox_preds.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_preds.view(
                        bbox_preds.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=num_instances,
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_preds[pos_inds].sum()
        return losses
