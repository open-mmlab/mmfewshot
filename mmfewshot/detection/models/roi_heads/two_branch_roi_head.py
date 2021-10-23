from typing import Dict, List, Tuple

import torch
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor


@HEADS.register_module()
class TwoBranchRoIHead(StandardRoIHead):
    """RoI head for `MPSR <https://arxiv.org/abs/2007.09384>`_."""

    def forward_auxiliary_train(self, feats: Tuple[Tensor],
                                gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for auxiliary data in training.

        Args:
            feats (tuple[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.
            gt_labels (list[Tensor]): List of class indices corresponding
                to each features, each is a 4D-tensor.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # bbox head forward and loss
        auxiliary_losses = self._bbox_forward_auxiliary_train(feats, gt_labels)
        return auxiliary_losses

    def _bbox_forward_auxiliary_train(self, feats: Tuple[Tensor],
                                      gt_labels: List[Tensor]) -> Dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            feats (tuple[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.
            gt_labels (list[Tensor]): List of class indices corresponding
                to each features, each is a 4D-tensor.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        cls_scores, = self.bbox_head.forward_auxiliary(feats)
        cls_score = torch.cat(cls_scores, dim=0)
        labels = torch.cat(gt_labels, dim=0)
        label_weights = torch.ones_like(labels)
        losses = self.bbox_head.auxiliary_loss(cls_score, labels,
                                               label_weights)

        return losses
