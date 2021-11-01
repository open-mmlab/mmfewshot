# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead
from torch import Tensor


@HEADS.register_module()
class NegMarginHead(ClsHead):
    """Classification head for `NegMargin <https://arxiv.org/abs/2003.12060>`_.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        temperature (float): Scaling factor of `cls_score`.
            Default: 30.0.
        margin (float): Margin of `cls_score`. Default: 0.0.
        metric_type (str): The way to calculate similarity.
            Options:['cosine', 'softmax']. Default: 'cosine'
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 temperature: float = 30.0,
                 margin: float = 0.0,
                 metric_type: str = 'cosine',
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} ' \
                                f'must be a positive integer'
        assert margin <= 0, f'margin = {margin} should <= 0'
        assert metric_type in ['cosine', 'softmax']
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.metric_type = metric_type
        self.temperature = temperature
        self.margin = margin
        self.init_layers()

    def init_layers(self) -> None:
        self.weight = nn.Parameter(
            torch.FloatTensor(self.num_classes, self.in_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward_train(self, x: Tensor, gt_label: Tensor, **kwargs) -> Dict:
        """Forward training data."""
        if self.metric_type == 'cosine':
            similarity = F.linear(F.normalize(x), F.normalize(self.weight))
        elif self.metric_type == 'softmax':
            similarity = F.linear(x, self.weight)
            similarity -= similarity.min(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f'metric type {self.metric_type} not supported')

        # for training and negative margin metric, add margin to the similarity
        one_hot_mask = torch.zeros((gt_label.size(0), self.num_classes),
                                   dtype=torch.uint8).to(gt_label.device)
        one_hot_mask = one_hot_mask.scatter_(1, gt_label.unsqueeze(1), 1)
        similarity = torch.where(one_hot_mask, similarity - self.margin,
                                 similarity)

        cls_score = similarity * self.temperature
        losses = self.loss(cls_score, gt_label)
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor, **kwargs) -> Dict:
        """Forward support data in meta testing."""
        return self.forward_train(x, gt_label, **kwargs)

    def forward_query(self, x: Tensor, **kwargs) -> List:
        """Forward query data in meta testing."""
        if self.metric_type == 'cosine':
            similarity = F.linear(F.normalize(x), F.normalize(self.weight))
        elif self.metric_type == 'softmax':
            similarity = F.linear(x, self.weight)
            similarity -= similarity.min(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f'metric type {self.metric_type} not supported')
        cls_score = similarity * self.temperature
        pred = F.softmax(cls_score, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        self.init_layers()
        self.train()

    def before_forward_query(self) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.eval()
