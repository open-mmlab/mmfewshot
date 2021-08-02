import torch
from mmdet.core import bbox2roi
from mmdet.models.builder import DETECTORS

from .query_support_detector import QuerySupportDetector


@DETECTORS.register_module()
class AttentionRPNDetector(QuerySupportDetector):
    """Implementation of `AttentionRPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        backbone (dict): Config of the backbone for query data.
        neck (dict | None): Config of the neck for query data and
            probably for support data. Default: None.
        support_backbone (dict | None): Config of the backbone for
            support data only. If None, support and query data will
            share same backbone. Default: None.
        support_neck (dict | None): Config of the neck for support
            data only. Default: None.
        rpn_head (dict | None): Config of rpn_head. Default: None.
        roi_head (dict | None): Config of roi_head. Default: None.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 support_backbone=None,
                 support_neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AttentionRPNDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'res4_roi_feats': [],
            'res5_roi_feats': []
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {}

    def extract_support_feat(self, img):
        """Extract features of support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of support images, each item with shape
                 (N, C, H, W).
        """
        feats = self.support_backbone(img)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        return feats

    def forward_model_init(self,
                           img,
                           img_metas,
                           gt_bboxes=None,
                           gt_labels=None,
                           **kwargs):
        """Extract and save support features for model initialization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.

        Returns:
            dict: A dict contains following keys:

                - `gt_labels` (Tensor): class indices corresponding to each
                    feature.
                - `res4_roi_feat` (Tensor): roi features of res4 layer.
                - `res5_roi_feat` (Tensor): roi features of res5 layer.
        """
        self.is_model_init = False
        # extract support template features will reset `is_model_init` flag
        assert gt_bboxes is not None and gt_labels is not None, \
            'forward support template require gt_bboxes and gt_labels.'
        assert len(gt_labels) == img.size(0), \
            'Support instance have more than two labels'

        feats = self.extract_support_feat(img)
        rois = bbox2roi([bboxes.unsqueeze(0) for bboxes in gt_bboxes])
        res4_roi_feat = self.rpn_head.extract_roi_feat(feats, rois)
        res5_roi_feat = self.roi_head.extract_roi_feat(feats, rois)
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['res4_roi_feats'].append(
            res4_roi_feat)
        self._forward_saved_support_dict['res5_roi_feats'].append(
            res5_roi_feat)

        return {
            'gt_labels': gt_labels,
            'res4_roi_feats': res4_roi_feat,
            'res5_roi_feats': res5_roi_feat
        }

    def model_init(self):
        """process the saved support features for model initialization."""
        self.inference_support_dict.clear()
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        res4_roi_feats = torch.cat(
            self._forward_saved_support_dict['res4_roi_feats'])
        res5_roi_feats = torch.cat(
            self._forward_saved_support_dict['res5_roi_feats'])
        class_ids = set(gt_labels.data.tolist())
        for class_id in class_ids:
            self.inference_support_dict[class_id] = {
                'res4_roi_feats':
                res4_roi_feats[gt_labels == class_id].mean([0, 2, 3], True),
                'res5_roi_feats':
                res5_roi_feats[gt_labels == class_id].mean([0], True)
            }
        # set the init flag
        self.is_model_init = True
        # clear support dict
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if (self.inference_support_dict == {}) or (not self.is_model_init):
            # process the saved support features
            self.model_init()

        results_dict = {}
        query_feats = self.extract_feat(img)
        for class_id in self.inference_support_dict.keys():
            support_res4_roi_feat = \
                self.inference_support_dict[class_id]['res4_roi_feats']
            support_res5_roi_feat = \
                self.inference_support_dict[class_id]['res5_roi_feats']
            if proposals is None:
                proposal_list = self.rpn_head.simple_test(
                    query_feats, support_res4_roi_feat, img_metas)
            else:
                proposal_list = proposals

            results_dict[class_id] = self.roi_head.simple_test(
                query_feats,
                support_res5_roi_feat,
                proposal_list,
                img_metas,
                rescale=rescale)
        results = [
            results_dict[i][0][0] for i in sorted(results_dict.keys())
            if len(results_dict[i])
        ]
        return [results]
