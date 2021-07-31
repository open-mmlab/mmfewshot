import torch
from mmdet.core import bbox2roi, bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class ContrastiveRoIHead(StandardRoIHead):
    """RoI head for `FSCE <https://arxiv.org/abs/2103.05950>`_."""

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing phase.

         Args:
             x (list[Tensor]): Features from the upstream network,
                each is a 4D-tensor.
             rois (Tensor): Shape of (num_proposals, 4) or (num_proposals, 5).

        Returns:
             dict[str, Tensor]: A dictionary of predicted results and output
                 features.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, contrast_feat = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
            contrast_feat=contrast_feat)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Forward function and calculate loss for box head in training phase.

        Args:
            x (list[Tensor]): Features from the upstream network,
                each is a 4D-tensor.
            sampling_results (obj:`SamplingResult`): Sampling result.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of output results and losses.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        proposal_ious = torch.cat(proposal_ious, dim=0)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        loss_contrast = self.bbox_head.loss_contrast(
            bbox_results['contrast_feat'],
            proposal_ious,
            labels=bbox_targets[0])
        loss_bbox.update(loss_contrast)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
