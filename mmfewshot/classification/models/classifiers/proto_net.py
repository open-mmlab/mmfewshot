# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .base_metric import BaseMetricClassifier


@CLASSIFIERS.register_module()
class ProtoNet(BaseMetricClassifier):
    """Implementation of `ProtoNet <https://arxiv.org/abs/1703.05175>`_."""

    def __init__(self,
                 head: Dict = dict(type='PrototypeHead'),
                 *args,
                 **kwargs) -> None:
        self.head_cfg = copy.deepcopy(head)
        super().__init__(head=head, *args, **kwargs)
