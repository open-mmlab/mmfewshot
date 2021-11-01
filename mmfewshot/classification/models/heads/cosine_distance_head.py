# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from torch import Tensor
from torch.nn.utils.weight_norm import WeightNorm

from .base_head import FewShotBaseHead


@HEADS.register_module()
class CosineDistanceHead(FewShotBaseHead):
    """Classification head for `Baseline++ https://arxiv.org/abs/2003.04390`_.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        temperature (float | None): Scaling factor of `cls_score`.
            Default: None.
        eps (float): Constant variable to avoid division by zero.
            Default: 0.00001.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 temperature: Optional[float] = None,
                 eps: float = 0.00001,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} ' \
                                f'must be a positive integer'

        self.in_channels = in_channels
        self.num_classes = num_classes
        if temperature is None:
            self.temperature = 2 if num_classes <= 200 else 10
        else:
            self.temperature = temperature
        self.eps = eps
        self.init_layers()

    def init_layers(self) -> None:
        self.fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
        self.fc = nn.utils.weight_norm(self.fc, name='weight', dim=0)

    def forward_train(self, x: Tensor, gt_label: Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + self.eps)
        cls_score = self.temperature * self.fc(x_normalized)
        losses = self.loss(cls_score, gt_label)
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor, **kwargs) -> Dict:
        """Forward support data in meta testing."""
        return self.forward_train(x, gt_label, **kwargs)

    def forward_query(self, x: Tensor, **kwargs) -> List:
        """Forward query data in meta testing."""
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + self.eps)
        cls_score = self.temperature * self.fc(x_normalized)
        pred = F.softmax(cls_score, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # the parameter will be random initialized for each task.
        self.init_layers()
        self.train()

    def before_forward_query(self) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.eval()

    def __deepcopy__(self, memo):
        """Deep copy function for nn.utils.weight_norm.

        During validation and meta testing, the whole module will be copied and
        the WeightNorm hook is not support deepcopy. More details in
        https://github.com/pytorch/pytorch/issues/28594.
        """
        # save and delete all weight_norm hook
        for module in self.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    delattr(module, hook.name)
        __deepcopy__ = self.__deepcopy__
        self.__deepcopy__ = None
        # deep copy itself
        result = copy.deepcopy(self)
        self.__deepcopy__ = __deepcopy__
        # restore all weight_norm hook
        for module in self.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    hook(module, None)
        return result
