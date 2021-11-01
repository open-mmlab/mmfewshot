# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from mmdet.models.builder import HEADS
from torch import Tensor

from .meta_rcnn_roi_head import MetaRCNNRoIHead


@HEADS.register_module()
class FSDetViewRoIHead(MetaRCNNRoIHead):
    """Roi head for `FSDetView <https://arxiv.org/abs/2007.12107>`_.

    Args:
        aggregation_layer (dict): Config of `aggregation_layer`.
            Default: None.
    """

    def __init__(self,
                 aggregation_layer: Optional[Dict] = None,
                 **kwargs) -> None:
        super().__init__(aggregation_layer=aggregation_layer, **kwargs)

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Roi features with shape (N, C).
            support_roi_feats (Tensor): Roi features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))
        roi_feats = torch.cat(roi_feats, dim=1)
        roi_feats = torch.cat((roi_feats, query_roi_feats), dim=1)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results
