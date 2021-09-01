import copy

from mmcls.models.builder import CLASSIFIERS

from .meta_metric_base import MetaMetricBaseClassifier


@CLASSIFIERS.register_module()
class MatchingNetClassifier(MetaMetricBaseClassifier):
    """Implementation of `MatchingNet  <https://arxiv.org/abs/1606.04080>`_."""

    def __init__(self, head=dict(type='MatchingHead'), *args, **kwargs):
        self.head_cfg = copy.deepcopy(head)
        super(MatchingNetClassifier, self).__init__(head=head, *args, **kwargs)
