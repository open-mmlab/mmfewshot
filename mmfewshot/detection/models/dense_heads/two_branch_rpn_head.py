# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import images_to_levels, multi_apply
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS
from torch import Tensor


@HEADS.register_module()
class TwoBranchRPNHead(RPNHead):
    """RPN head for `MPSR <https://arxiv.org/abs/2007.09384>`_.

    Args:
        mid_channels (int): Input channels of `rpn_cls_conv`. Default: 64.
    """

    def __init__(self, mid_channels: int = 64, **kwargs) -> None:
        self.mid_channels = mid_channels
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls_conv = nn.Conv2d(self.feat_channels,
                                      self.num_anchors * self.mid_channels, 1)
        self.rpn_cls = nn.Conv2d(self.mid_channels, self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def forward_single(self, feat: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature map of a single scale level."""
        feat = self.rpn_conv(feat)
        feat = F.relu(feat, inplace=True)
        cls_feat = self.rpn_cls_conv(feat)
        b, _, h, w = cls_feat.size()
        cls_feat = cls_feat.view(b * self.num_anchors, self.mid_channels, h, w)
        rpn_cls_score = self.rpn_cls(cls_feat).view(-1, self.num_anchors, h, w)
        rpn_bbox_pred = self.rpn_reg(feat)
        return rpn_cls_score, rpn_bbox_pred

    def forward_auxiliary_single(self, feat: Tensor) -> Tuple[Tensor, ]:
        """Forward auxiliary feature map of a single scale level."""
        feat = self.rpn_conv(feat)
        feat = F.relu(feat, inplace=True)
        # only use center 2x2(last 3x3)
        feat = feat[:, :, 3:-3, 3:-3]
        cls_feat = self.rpn_cls_conv(feat)
        b, _, h, w = cls_feat.size()
        cls_feat = cls_feat.view(b * self.num_anchors, self.mid_channels, h, w)
        rpn_cls_score = self.rpn_cls(cls_feat).view(-1, self.num_anchors, h, w)
        return rpn_cls_score,  # return a tuple for multi_apply

    def forward_auxiliary(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward auxiliary features at multiple scales.

        Args:
            feats (list[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.

        Returns:
            list[Tensor]: Classification scores for all scale levels, each is
                a 4D-tensor, the channels number is num_anchors * num_classes.
        """
        return multi_apply(self.forward_auxiliary_single, feats)

    def forward_train(self,
                      x: List[Tensor],
                      auxiliary_rpn_feats: List[Tensor],
                      img_metas: List[Dict],
                      gt_bboxes: List[Tensor],
                      gt_labels: Optional[List[Tensor]] = None,
                      gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs) -> Tuple[Dict, List[Tensor]]:
        """
        Args:
            x (list[Tensor]): Features from FPN, each item with shape
                (N, C, H, W).
            auxiliary_rpn_feats (list[Tensor]): Auxiliary features
                from FPN, each item with shape (N, C, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,). Default: None.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.
            proposal_cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (List[Tensor]): Proposals of each image.
        """
        outs = self(x)
        auxiliary_cls_scores = self.forward_auxiliary(auxiliary_rpn_feats)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore) + auxiliary_cls_scores
        losses = self.loss(*loss_inputs)
        proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    def loss_bbox_single(self, bbox_pred: Tensor, anchors: Tensor,
                         bbox_targets: Tensor, bbox_weights: Tensor,
                         num_total_samples: int) -> Tuple[Dict, ]:
        """Compute loss of a single scale level.

        Args:
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple[dict[str, Tensor]]: A dictionary of loss components.
        """

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_bbox,  # return a tuple for multi_apply

    def loss_cls_single(self, cls_score: Tensor, labels: Tensor,
                        label_weights: Tensor,
                        num_total_samples: int) -> Tuple[Dict, ]:
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple[dict[str, Tensor]]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.flatten()
        label_weights = label_weights.flatten()
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        return loss_cls,  # return a tuple for multi_apply

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'auxiliary_cls_scores'))
    def loss(self,
             cls_scores: List[Tensor],
             bbox_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             gt_labels: List[Tensor],
             img_metas: List[Dict],
             gt_bboxes_ignore: Optional[List[Tensor]] = None,
             auxiliary_cls_scores: Optional[List[Tensor]] = None) -> Dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level, each
                item with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss. Default: None.
            auxiliary_cls_scores (list[Tensor] | None): Box scores for each
                scale level, each item with shape (N, num_anchors *
                num_classes, H, W). Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        # prepare the anchors and training targets for main data stream
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples_main = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # prepare the training targets (classification only)
        # for auxiliary data stream
        auxiliary_labels_list = [
            torch.zeros_like(cls_score, dtype=torch.long).to(device)
            for cls_score in auxiliary_cls_scores
        ]
        auxiliary_label_weights_list = [
            torch.ones_like(cls_score, dtype=torch.float).to(device)
            for cls_score in auxiliary_cls_scores
        ]
        num_total_samples_cls = num_total_samples_main + sum(
            label_weights.numel()
            for label_weights in auxiliary_label_weights_list)

        # get loss
        losses_cls, = multi_apply(
            self.loss_cls_single,
            cls_scores,
            labels_list,
            label_weights_list,
            num_total_samples=num_total_samples_cls,
        )
        losses_cls_auxiliary, = multi_apply(
            self.loss_cls_single,
            auxiliary_cls_scores,
            auxiliary_labels_list,
            auxiliary_label_weights_list,
            num_total_samples=num_total_samples_cls,
        )
        losses_bbox, = multi_apply(
            self.loss_bbox_single,
            bbox_preds,
            all_anchor_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples_main,
        )

        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_cls_auxiliary=losses_cls_auxiliary,
            loss_rpn_bbox=losses_bbox)

    def _get_bboxes_single(self,
                           cls_scores: List[Tensor],
                           bbox_preds: List[Tensor],
                           mlvl_anchors: List[Tensor],
                           img_shape: Tuple[int],
                           scale_factor: np.ndarray,
                           cfg: ConfigDict,
                           rescale: bool = False) -> Tensor:
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (np.ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_proposals = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.flatten()
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]

            if 0 < cfg.nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_shape)
            # following the original implementation of MPSR based on
            # maskrcnn benchmark, we perform nms on each level separately
            _, keep = batched_nms(
                proposals, scores,
                scores.new_full((scores.size(0), ), 0, dtype=torch.long),
                cfg.nms)
            mlvl_scores.append(scores[keep])
            mlvl_bbox_proposals.append(proposals[keep])
            mlvl_valid_anchors.append(anchors[keep])
            level_ids.append(scores[keep].new_full((scores[keep].size(0), ),
                                                   idx,
                                                   dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        proposals = torch.cat(mlvl_bbox_proposals)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        assert (cfg.get('max_per_batch', None) is not None) ^ \
               (cfg.get('max_per_img', None) is not None), \
               'max_per_batch and max_per_img can not be set at the same time.'
        # NOTE: We add max_per_batch to be consistent with the original
        # implementation of MPSR based on maskrcnn benchmark.
        # it will return all the proposals and scores for each image
        # and then select Top k proposals from a batch of proposals
        if self.training and cfg.get('max_per_batch', None) is not None:
            return torch.cat((proposals, scores.unsqueeze(1)), dim=1)
        # return Top k proposals according to max_per_img for each image
        elif proposals.numel() > 0:
            ranked_scores, rank_inds = scores.sort(descending=True)
            topk_inds = rank_inds[:cfg.max_per_img]
            dets = proposals[topk_inds, :]
            return dets
        else:
            return proposals.new_zeros(0, 5)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores: List[Tensor],
                   bbox_preds: List[Tensor],
                   img_metas: List[Dict],
                   cfg: Optional[ConfigDict] = None,
                   rescale: bool = False,
                   with_nms: bool = True) -> List[Tensor]:
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (ConfigDict | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            List[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)

        # NOTE: We add max_per_batch to be consistent with the original
        # implementation of MPSR based on maskrcnn benchmark.

        # There are two arguments to control the max number of proposals of
        # a batch in two_branch_rpn_head.py: max_per_batch and max_per_img.
        # max_per_batch only works in the training phase and it is applied
        # to filter proposals of the whole batch.
        # max_per_img is used to filter the proposals from each image
        # respectively.
        if self.training and cfg.get('max_per_batch', None) is not None:
            batch_scores = torch.cat(result_list)[:, -1]
            num_proposals = [proposals.size(0) for proposals in result_list]
            post_nms_top_n = min(cfg.max_per_batch, batch_scores.size()[0])
            _, indices_sorted = batch_scores.sort(descending=True)
            indices_mask = torch.zeros_like(batch_scores, dtype=torch.bool)
            indices_mask[indices_sorted[:post_nms_top_n]] = 1
            indices_mask = indices_mask.split(num_proposals)
            for img_id in range(len(img_metas)):
                result_list[img_id] = result_list[img_id][indices_mask[img_id]]
        return result_list
