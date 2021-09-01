import torch
from mmdet.models.builder import HEADS

from .meta_rcnn_roi_head import MetaRCNNRoIHead


@HEADS.register_module()
class FSDetViewRoIHead(MetaRCNNRoIHead):
    """Roi head for `FSDetView <https://arxiv.org/abs/1908.01998>`_.

    Args:
        aggregation_layer (dict): Config of `aggregation_layer`.
            Default: None.
    """

    def __init__(self, aggregation_layer=None, **kwargs):
        super(FSDetViewRoIHead, self).__init__(
            aggregation_layer=aggregation_layer, **kwargs)

    def _bbox_forward(self, query_rois_feats, support_rois_feats):
        """Box head forward function used in both training and testing.

        Args:
            query_rois_feats (Tensor): Roi features with shape (N, C).
            support_rois_feats (Tensor): Roi features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        rois_feats = self.aggregation_layer(
            query_feat=query_rois_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_rois_feats.view(1, -1, 1, 1))
        rois_feats = torch.cat(rois_feats, dim=1)
        rois_feats = torch.cat((rois_feats, query_rois_feats), dim=1)
        cls_score, bbox_pred = self.bbox_head(rois_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results
