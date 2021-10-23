import copy
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi, images_to_levels, multi_apply
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS, build_roi_extractor
from torch import Tensor

from mmfewshot.detection.models.utils import build_aggregator


@HEADS.register_module()
class AttentionRPNHead(RPNHead):
    """RPN head for `Attention RPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        num_support_ways (int): Number of sampled classes (pos + neg).
        num_support_shots (int): Number of shot for each classes.
        aggregation_layer (dict): Config of `aggregation_layer`.
        roi_extractor (dict): Config of `roi_extractor`.
    """

    def __init__(self,
                 num_support_ways: int,
                 num_support_shots: int,
                 aggregation_layer: Dict = dict(
                     type='AggregationLayer',
                     aggregator_cfgs=[
                         dict(
                             type='DepthWiseCorrelationAggregator',
                             in_channels=1024,
                             with_fc=False)
                     ]),
                 roi_extractor: Dict = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=0),
                     out_channels=1024,
                     featmap_strides=[16]),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        assert roi_extractor is not None, \
            'missing config of roi_extractor.'
        assert aggregation_layer is not None, \
            'missing config of aggregation_layer.'
        self.aggregation_layer = \
            build_aggregator(copy.deepcopy(aggregation_layer))
        self.roi_extractor = \
            build_roi_extractor(copy.deepcopy(roi_extractor))

    def extract_roi_feat(self, feats: List[Tensor], rois: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (list[Tensor]): Input features with shape (N, C, H, W).
            rois (Tensor): with shape (m, 5).

         Returns:
            Tensor: RoI features with shape (N, C, H, W).
        """
        return self.roi_extractor(feats, rois)

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      query_gt_bboxes: List[Tensor],
                      query_img_metas: List[Dict],
                      support_gt_bboxes: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs) -> Tuple[Dict, List[Tensor]]:
        """Forward function in training phase.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W)..
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            query_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                query image, each item with shape (num_gts, 4).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            support_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                support image, each item with shape (num_gts, 4).
            query_gt_bboxes_ignore (list[Tensor]): List of ground truth bboxes
                to be ignored of query image with shape (num_ignored_gts, 4).
                Default: None.
            proposal_cfg (:obj:`ConfigDict`): Test / postprocessing
                configuration. if None, test_cfg would be used. Default: None.

        Returns:
            tuple: loss components and proposals of each image.

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - proposal_list (list[Tensor]): Proposals of each image.
        """
        query_feat = query_feats[0]
        support_rois = bbox2roi([bboxes for bboxes in support_gt_bboxes])
        support_roi_feats = self.extract_roi_feat(support_feats, support_rois)
        # Support features are placed in follow order:
        # [pos, neg, ..., pos, neg] * batch size
        avg_support_feats = [
            support_roi_feats[i * self.num_support_shots:(i + 1) *
                              self.num_support_shots].mean([0, 2, 3],
                                                           keepdim=True)
            for i in range(
                support_roi_feats.size(0) // self.num_support_shots)
        ]
        # Concat all positive pair
        pos_pair_feats = [
            self.aggregation_layer(
                query_feat=query_feat[i].unsqueeze(0),
                support_feat=avg_support_feats[i * self.num_support_ways])[0]
            for i in range(query_feat.size(0))
        ]
        # Concat all negative pair
        neg_pair_feats = [
            self.aggregation_layer(
                query_feat=query_feat[i].unsqueeze(0),
                support_feat=avg_support_feats[i * self.num_support_ways + j +
                                               1])[0]
            for i in range(query_feat.size(0))
            for j in range(self.num_support_ways - 1)
        ]

        batch_size = len(query_img_metas)
        # pair_flags will set all the gt_label to bg classes in losses
        pair_flags = [1 for _ in range(batch_size)]
        repeat_query_img_metas = copy.deepcopy(query_img_metas)
        repeat_query_gt_bboxes = copy.deepcopy(query_gt_bboxes)
        for i in range(batch_size):
            repeat_query_img_metas.extend([query_img_metas[i]] *
                                          (self.num_support_ways - 1))
            repeat_query_gt_bboxes.extend([query_gt_bboxes[i]] *
                                          (self.num_support_ways - 1))
            pair_flags.extend([0] * (self.num_support_ways - 1))
        outs = self([torch.cat(pos_pair_feats + neg_pair_feats)])
        loss_inputs = outs + (repeat_query_gt_bboxes, repeat_query_img_metas)
        losses = self.loss(
            *loss_inputs,
            gt_bboxes_ignore=query_gt_bboxes_ignore,
            pair_flags=pair_flags)
        proposal_list = self.get_bboxes(
            *outs, repeat_query_img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores: List[Tensor],
             bbox_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             img_metas: List[Dict],
             gt_labels: Optional[List[Tensor]] = None,
             gt_bboxes_ignore: Optional[List[Tensor]] = None,
             pair_flags: Optional[List[bool]] = None) -> Dict:
        """Compute losses of rpn head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                Default: None.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None
            pair_flags (list[bool]): Indicate predicted result is from positive
                pair or negative pair with shape (N). Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

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
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # flip neg support labels
        neg_idx = [f == 0 for f in pair_flags]
        flip_neg_index = []
        for lvl in range(len(labels_list)):
            flip_neg_index += (labels_list[lvl][neg_idx] == 0)
            labels_list[lvl][neg_idx] = 1
            bbox_weights_list[lvl][neg_idx] = 0
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos -
            flip_neg_index.sum())

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single Tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feat: Tensor,
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[Tensor]:
        """Test function without test time augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each item with
                shape(N, C, H, W).
            support_feat (Tensor): Support features with shape (N, C, H, W).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results.
                Default: False.

        Returns:
            List[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        feats = self.aggregation_layer(
            query_feat=query_feats[0], support_feat=support_feat)
        proposal_list = self.simple_test_rpn(feats, query_img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, query_img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        return proposal_list
