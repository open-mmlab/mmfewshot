# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .base_metric import BaseMetricClassifier


@CLASSIFIERS.register_module()
class RelationNet(BaseMetricClassifier):
    """Implementation of `RelationNet <https://arxiv.org/abs/1711.06025>`_."""

    def __init__(self,
                 head: Dict = dict(
                     type='RelationHead',
                     in_channels=64,
                     feature_size=(19, 19)),
                 *args,
                 **kwargs) -> None:
        self.head_cfg = copy.deepcopy(head)
        super().__init__(head=head, *args, **kwargs)
