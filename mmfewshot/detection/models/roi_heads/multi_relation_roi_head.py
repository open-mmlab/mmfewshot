# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor


@HEADS.register_module()
class MultiRelationRoIHead(StandardRoIHead):
    """Roi head for `AttentionRPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        num_support_ways (int): Number of sampled classes (pos + neg).
        num_support_shots (int): Number of shot for each classes.
        sample_fractions (Sequence[int | float]):
            Fractions of positive samples, negative samples from positive
            pair, negative samples from negative pair. Default: (1, 2, 1).
    """

    def __init__(self,
                 num_support_ways: int = 2,
                 num_support_shots: int = 5,
                 sample_fractions: Sequence[Union[float, int]] = (1, 2, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        self.sample_fractions = sample_fractions

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_bboxes: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """All arguments excepted proposals are passed in tuple of (query,
        support).

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative query-support pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each bbox from query image.
            support_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                support image with shape (num_gts, 4) in
                [tl_x, tl_y, br_x, br_y] format.
            query_gt_bboxes_ignore (None | list[Tensor]): Specify which
                bounding boxes from query image can be ignored when computing
                the loss. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # prepare contrastive training data
        batch_size = len(query_img_metas)
        repeat_query_feats = []
        for lvl in range(len(query_feats)):
            repeat_query_feats.append([query_feats[lvl]])
        for i in range(batch_size):
            query_gt_labels[i] = torch.zeros_like(query_gt_labels[i])
            query_gt_labels.extend([torch.zeros_like(query_gt_labels[i])] *
                                   (self.num_support_ways - 1))
            for lvl in range(len(query_feats)):
                repeat_query_feats[lvl].append(
                    query_feats[lvl][i].unsqueeze(0).repeat(
                        (self.num_support_ways - 1, 1, 1, 1)))
            query_img_metas.extend([query_img_metas[i]] *
                                   (self.num_support_ways - 1))
            query_gt_bboxes.extend([query_gt_bboxes[i]] *
                                   (self.num_support_ways - 1))
        for lvl in range(len(repeat_query_feats)):
            repeat_query_feats[lvl] = torch.cat(repeat_query_feats[lvl])

        sampling_results = []
        # assign gts and sample proposals
        if self.with_bbox:
            num_imgs = len(query_img_metas)
            assert num_imgs == len(proposals), \
                'proposals should keep same length with batch_size*(pos+neg).'
            if query_gt_bboxes_ignore is None:
                query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposals[i], query_gt_bboxes[i],
                    query_gt_bboxes_ignore[i], query_gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposals[i],
                    query_gt_bboxes[i],
                    query_gt_labels[i],
                    feats=[
                        lvl_feat[i][None] for lvl_feat in repeat_query_feats
                    ])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                batch_size, repeat_query_feats, support_feats,
                sampling_results, query_gt_bboxes, query_gt_labels,
                support_gt_bboxes)
            if bbox_results is not None:
                losses.update(bbox_results['loss_bbox'])

        return losses

    def extract_roi_feat(self, feats: List[Tensor], rois: Tensor) -> Tensor:
        """Extract BBOX feature function used in both training and testing.

        Args:
            feats (list[Tensor]): Features from backbone, each item with shape
                (N, C, W, H).
            rois (Tensor): shape (num_proposals, 5).

        Returns:
            Tensor: Roi features with shape (num_proposals, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Roi features with shape (N*K, C).
            support_roi_feats (Tensor): Roi features with shape (N, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        cls_score, bbox_pred = [], []
        batch_size = len(support_roi_feats)
        for query_rois_feat, support_rois_feat in zip(
                torch.chunk(query_roi_feats, batch_size), support_roi_feats):
            cls_score_single, bbox_pred_single = self.bbox_head(
                query_rois_feat, support_rois_feat)
            cls_score.append(cls_score_single)
            bbox_pred.append(bbox_pred_single)

        bbox_results = dict(
            cls_score=torch.cat(cls_score), bbox_pred=torch.cat(bbox_pred))
        return bbox_results

    def _bbox_forward_train(self, batch_size: int, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_bboxes: List[Tensor]) -> Dict:
        """Forward function and calculate loss for bbox head in training.

        Args:
            batch_size (int): Batch size.
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding
                to each bbox for query images.
            support_gt_bboxes (list[Tensor]): Ground truth bboxes for support
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_roi_feat(query_feats, query_rois)
        support_rois = bbox2roi([bboxes for bboxes in support_gt_bboxes])
        support_roi_feats = self.extract_roi_feat(support_feats, support_rois)
        avg_support_roi_feats = [
            support_roi_feats[i * self.num_support_shots:(i + 1) *
                              self.num_support_shots].mean(0, True)
            for i in range(
                support_roi_feats.size(0) // self.num_support_shots)
        ]
        if batch_size > 1:
            pos_avg_support_roi_feats = []
            neg_avg_support_roi_feats = []
            for b in range(batch_size):
                start = b * self.num_support_ways
                end = (b + 1) * self.num_support_ways
                pos_avg_support_roi_feats.extend(
                    avg_support_roi_feats[start:start + 1])
                neg_avg_support_roi_feats.extend(avg_support_roi_feats[start +
                                                                       1:end])
            avg_support_roi_feats = \
                pos_avg_support_roi_feats + neg_avg_support_roi_feats

        bbox_results = self._bbox_forward(query_roi_feats,
                                          avg_support_roi_feats)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets

        # flip negative pair labels
        num_sampler_per_image = sampling_results[0].bboxes.size(0)
        num_pos_pair_samples = batch_size * num_sampler_per_image
        labels[num_pos_pair_samples:] = 1
        bbox_weights[num_pos_pair_samples:] = 0

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            query_rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_pos_pair_samples,
            sample_fractions=self.sample_fractions)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feat: Tensor,
                    proposals: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each
                item with shape (N, C, H, W).
            support_feat (Tensor): Support features with shape (N, C, H, W).
            proposals (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (Tensor or list[Tensor]): Region proposals.
                Default: None.
            rescale (bool): Whether to rescale the results. Default: False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feat,
            query_img_metas,
            proposals,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feat: Tensor,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feat (Tensor): Support feature with shape (N, C, H, W).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[Tensor, Tensor]: BBox of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        rois = bbox2roi(proposals)
        query_roi_feats = self.extract_roi_feat(query_feats, rois)
        bbox_results = self._bbox_forward(query_roi_feats, [support_feat])
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
