# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
from torch import Tensor


@DETECTORS.register_module()
class MPSR(TwoStageDetector):
    """Implementation of `MPSR. <https://arxiv.org/abs/2007.09384>`_.

    Args:
        rpn_select_levels (list[int]): Specify the corresponding
            level of fpn features for each scale of image. The selected
            features will be fed into rpn head.
        roi_select_levels (list[int]): Specific which level of fpn
            features to be selected for each scale of image. The selected
            features will be fed into roi head.
    """

    def __init__(self, rpn_select_levels: List[int],
                 roi_select_levels: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert rpn_select_levels, 'rpn_select_levels can not be None.'
        assert roi_select_levels, 'roi_select_levels can not be None.'
        assert len(rpn_select_levels) == len(roi_select_levels), \
            'lengths of rpn_select_levels and roi_select_levels mismatch.'
        self.rpn_select_levels = rpn_select_levels
        self.roi_select_levels = roi_select_levels
        self.num_fpn_levels = max(
            max(rpn_select_levels), max(roi_select_levels)) + 1

    def extract_auxiliary_feat(
            self, auxiliary_data_list: List[Dict]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract and select features from data list at multiple scale.

        Args:
            auxiliary_data_list (list[dict]): List of data at different
                scales. In most cases, each dict contains: `img`, `img_metas`,
                `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.

        Returns:
            tuple:
                rpn_feats (list[Tensor]): Features at multiple scale used
                    for rpn head training.
                roi_feats (list[Tensor]): Features at multiple scale used
                    for roi head training.
        """

        rpn_feats = []
        roi_feats = []
        for scale, data in enumerate(auxiliary_data_list):
            feats = self.backbone(data['img'])
            if self.with_neck:
                feats = self.neck(feats)
            assert len(feats) >= self.num_fpn_levels, \
                f'minimum number of fpn levels is {self.num_fpn_levels}.'
            # for each scale of image, only one level of fpn features will be
            # selected for training.
            if scale == 5:
                # 13 x 13 -> 9 x 9
                rpn_feats.append(feats[self.rpn_select_levels[scale]][:, :,
                                                                      2:-2,
                                                                      2:-2])
            else:
                rpn_feats.append(feats[self.rpn_select_levels[scale]])
            roi_feats.append(feats[self.roi_select_levels[scale]])
        return rpn_feats, roi_feats

    def forward_train(self, main_data: Dict, auxiliary_data_list: List[Dict],
                      **kwargs) -> Dict:
        """
        Args:
            main_data (dict): In most cases, dict of main data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            auxiliary_data_list (list[dict]): List of data at different
                scales. In most cases, each dict contains: `img`, `img_metas`,
                `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # train model with regular pipeline
        x = self.extract_feat(main_data['img'])
        # train model with refine pipeline
        auxiliary_rpn_feats, auxiliary_roi_feats = \
            self.extract_auxiliary_feat(auxiliary_data_list)

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            auxiliary_rpn_feats,
            main_data['img_metas'],
            main_data['gt_bboxes'],
            gt_labels=None,
            gt_bboxes_ignore=main_data.get('gt_bboxes_ignore', None),
            proposal_cfg=proposal_cfg)

        roi_losses = self.roi_head.forward_train(
            x, main_data['img_metas'], proposal_list,
            main_data['gt_bboxes'], main_data['gt_labels'],
            main_data.get('gt_bboxes_ignore', None), **kwargs)

        auxiliary_roi_losses = self.roi_head.forward_auxiliary_train(
            auxiliary_roi_feats,
            [torch.cat(data['gt_labels']) for data in auxiliary_data_list])

        losses = dict()
        losses.update(rpn_losses)
        losses.update(roi_losses)
        losses.update(auxiliary_roi_losses)

        return losses

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                main_data: Dict = None,
                auxiliary_data: Dict = None,
                img: List[Tensor] = None,
                img_metas: List[Dict] = None,
                return_loss: bool = True,
                **kwargs) -> Dict:
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, the input will be main and auxiliary data
        for training., and when ``resturn_loss=False``, the input will be
        img and img_meta for testing.

        Args:
            main_data (dict): Used for :func:`forward_train`. Dict of
                data and data info, where each dict has: `img`, `img_metas`,
                `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`. Default: None.
            auxiliary_data (dict): Used for :func:`forward_train`. Dict of
                data and data info at multiple scales, where each key use
                different suffix to indicate different scale. For example,
                `img_scale_i`, `img_metas_scale_i`, `gt_bboxes_scale_i`,
                `gt_labels_scale_i`, `gt_bboxes_ignore_scale_i`, where
                `i` in range of 0 to number of scales. Default: None.
            img (list[Tensor]): Used for func:`forward_test` or
                :func:`forward_model_init`. List of tensors of shape
                (1, C, H, W). Typically these should be mean centered
                and std scaled. Default: None.
            img_metas (list[dict]): Used for func:`forward_test` or
                :func:`forward_model_init`.  List of image info dict
                where each dict has: `img_shape`, `scale_factor`, `flip`,
                and may also contain `filename`, `ori_shape`, `pad_shape`,
                and `img_norm_cfg`. For details on the values of these keys,
                see :class:`mmdet.datasets.pipelines.Collect`. Default: None.
            return_loss (bool): If set True call :func:`forward_train`,
                otherwise call :func:`forward_test`. Default: True.
        """
        if return_loss:
            # collect data or data info at same scale into one dict
            keys = list(auxiliary_data.keys())
            num_scales = max(map(int, [key[-1] for key in keys])) + 1
            auxiliary_data_list = [{
                key.replace(f'_scale_{scale}', ''): auxiliary_data[key]
                for key in keys if f'_scale_{scale}' in key
            } for scale in range(num_scales)]
            return self.forward_train(main_data, auxiliary_data_list, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data: Dict, optimizer: Union[object, Dict]) -> Dict:
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['main_data']['img_metas']))

        return outputs

    def val_step(self,
                 data: Dict,
                 optimizer: Optional[Union[object, Dict]] = None) -> Dict:
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['main_data']['img_metas']))

        return outputs
