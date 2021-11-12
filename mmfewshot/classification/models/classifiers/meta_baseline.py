# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .base_metric import BaseMetricClassifier


@CLASSIFIERS.register_module()
class MetaBaseline(BaseMetricClassifier):
    """Implementation of `MetaBaseline <https://arxiv.org/abs/2003.04390>`_.

    Args:
        head (dict): Config of classification head for training.
    """

    def __init__(self,
                 head: Dict = dict(type='MetaBaselineHead'),
                 *args,
                 **kwargs) -> None:
        super().__init__(head=head, *args, **kwargs)
