import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS, build_neck
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor


@HEADS.register_module()
class MetaRCNNRoIHead(StandardRoIHead):
    """Roi head for `MetaRCNN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        aggregation_layer (ConfigDict): Config of `aggregation_layer`.
            Default: None.
    """

    def __init__(self,
                 aggregation_layer: Optional[ConfigDict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert aggregation_layer is not None, \
            'missing config of `aggregation_layer`.'
        self.aggregation_layer = build_neck(copy.deepcopy(aggregation_layer))

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Tensor],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).
            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        sampling_results = []
        if self.with_bbox:
            num_imgs = len(query_img_metas)
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
                    feats=[lvl_feat[i][None] for lvl_feat in query_feats])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                query_feats, support_feats, sampling_results, query_img_metas,
                query_gt_bboxes, query_gt_labels, support_gt_labels)
            if bbox_results is not None:
                losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_labels[img_id].size(0)))
            random_query_label = query_gt_labels[img_id][random_index]
            for i in range(support_feat.size(0)):
                # following the official code each image only sample first
                # class for training. The official code only use the first
                # `query_gt_labels` for sampling support class, while we
                # random one from `query_gt_labels` instead.
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat[i].unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size

        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        """Extracting query BBOX features, which is used in both training and
        testing.

        Args:
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (m, 5).

        Returns:
            Tensor: RoI features with shape (N, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.

        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).

        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        if self.with_shared_head:
            for lvl in range(len(feats)):
                out.append(self.shared_head(feats[lvl], is_support=True))
        else:
            out = feats
        return out

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1))
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feats_dict,
            query_img_metas,
            proposal_list,
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
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
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
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            # the official code use the first class background score as final
            # background score, while we use average of all classes'
            # background scores instead.
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[cls_scores_dict.keys()[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[bbox_preds_dict.keys()[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # split batch bbox prediction back to each image
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
