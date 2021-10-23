from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weighted_loss
from torch import Tensor
from typing_extensions import Literal


@weighted_loss
def nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Wrapper of nll loss."""
    return F.nll_loss(pred, target, reduction='none')


@LOSSES.register_module()
class NLLLoss(nn.Module):
    """NLLLoss.

    Args:
        reduction (str): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum". Default: 'mean'.
        loss_weight (float): The weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[float, int]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function of loss.

        Args:
            pred (Tensor): The prediction with shape (N, C).
            target (Tensor): The learning target of the prediction.
                with shape (N, 1).
            weight (Tensor | None): Weight of the loss for each
                prediction. Default: None.
            avg_factor (float | int | None): Average factor that is used to
                average the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * nll_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
