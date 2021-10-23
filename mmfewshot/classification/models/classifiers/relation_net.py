import copy
from typing import Dict

from mmcls.models.builder import CLASSIFIERS

from .meta_metric_base import MetaMetricBaseClassifier


@CLASSIFIERS.register_module()
class RelationNetClassifier(MetaMetricBaseClassifier):
    """Implementation of `RelationNet  <https://arxiv.org/abs/1711.06025>`_."""

    def __init__(self,
                 head: Dict = dict(type='RelationHead', in_channels=64),
                 *args,
                 **kwargs) -> None:
        self.head_cfg = copy.deepcopy(head)
        super().__init__(head=head, *args, **kwargs)
