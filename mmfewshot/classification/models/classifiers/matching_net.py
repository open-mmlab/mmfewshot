# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .base_metric import BaseMetricClassifier


@CLASSIFIERS.register_module()
class MatchingNet(BaseMetricClassifier):
    """Implementation of `MatchingNet <https://arxiv.org/abs/1606.04080>`_."""

    def __init__(self, head: Dict = dict(type='MatchingHead'), *args,
                 **kwargs) -> None:
        self.head_cfg = copy.deepcopy(head)
        super().__init__(head=head, *args, **kwargs)
