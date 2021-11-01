# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .finetune_base import FinetuneBaseClassifier


@CLASSIFIERS.register_module()
class NegMarginClassifier(FinetuneBaseClassifier):
    """Implementation of `NegMargin  <https://arxiv.org/abs/2003.12060>`_."""

    def __init__(self,
                 head: Dict = dict(
                     type='NegMarginHead',
                     metric_type='cosine',
                     num_classes=100,
                     in_channels=1600,
                     margin=-0.02,
                     temperature=30.0),
                 meta_test_head: Dict = dict(
                     type='NegMarginHead',
                     metric_type='cosine',
                     num_classes=5,
                     in_channels=1600,
                     margin=0.0,
                     temperature=5.0),
                 *args,
                 **kwargs) -> None:
        super().__init__(
            head=head, meta_test_head=meta_test_head, *args, **kwargs)
