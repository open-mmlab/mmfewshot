from typing import Optional

import torch
import torch.nn as nn
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from torch import Tensor
from typing_extensions import Literal


@LOSSES.register_module()
class SupervisedContrastiveLoss(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                features: Tensor,
                labels: Tensor,
                ious: Tensor,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = self.loss_weight
        if decay_rate is not None:
            loss_weight = self.loss_weight * decay_rate

        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask with shape [N, N], mask_{i, j}=1
        # if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().to(features.device)

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask *
                              label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
        if keep.sum() == 0:
            # return zero loss
            return per_label_log_prob.sum() * 0
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coefficient = self.reweight_func(ious)
        coefficient = coefficient[keep]
        if weight is not None:
            weight = weight[keep]
        loss = loss * coefficient
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss_weight * loss

    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay
