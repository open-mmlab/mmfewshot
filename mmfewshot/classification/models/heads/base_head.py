# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

from mmcls.models.builder import HEADS, build_loss
from mmcls.models.losses import Accuracy
from mmcv.runner import BaseModule
from torch import Tensor


@HEADS.register_module()
class BaseFewShotHead(BaseModule, metaclass=ABCMeta):
    """Base head for few shot classifier.

    Args:
        loss (dict): Training loss.
        topk (tuple[int]): Topk metric for computing the accuracy.
        cal_acc (bool): Whether to compute the accuracy during training.
            Default: False.
    """

    def __init__(self,
                 loss: Dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk: Tuple[int] = (1, ),
                 cal_acc: bool = False) -> None:
        super().__init__()
        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score: Tensor, gt_label: Tensor) -> Dict:
        """Calculate loss.

        Args:
            cls_score (Tensor): The prediction.
            gt_label (Tensor): The learning target of the prediction.

        Returns:
            Dict: The calculated loss.
        """
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    @abstractmethod
    def forward_train(self, **kwargs):
        """Forward training data."""

    @abstractmethod
    def forward_support(self, x, gt_label, **kwargs):
        """Forward support data in meta testing."""

    @abstractmethod
    def forward_query(self, x, **kwargs):
        """Forward query data in meta testing."""

    @abstractmethod
    def before_forward_support(self):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """

    @abstractmethod
    def before_forward_query(self):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
