from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .finetune_base import FinetuneBaseClassifier


@CLASSIFIERS.register_module()
class BaselinePlusClassifier(FinetuneBaseClassifier):
    """Implementation of `Baseline++ <https://arxiv.org/abs/1904.04232>`_.

    Args:
        head (dict): Config of classification head for training.
        meta_test_head (dict): Config of classification head for meta testing.
            the `meta_test_head` only will be built and run in meta testing.
    """

    def __init__(self,
                 head: Dict = dict(
                     type='CosineDistanceHead',
                     num_classes=100,
                     in_channels=512),
                 meta_test_head: Dict = dict(
                     type='CosineDistanceHead', num_classes=5,
                     in_channels=512),
                 *args,
                 **kwargs) -> None:
        super().__init__(
            head=head, meta_test_head=meta_test_head, *args, **kwargs)
