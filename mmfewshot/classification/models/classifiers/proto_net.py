import copy

from mmcls.models.builder import CLASSIFIERS

from .meta_metric_base import MetaMetricBaseClassifier


@CLASSIFIERS.register_module()
class ProtoNetClassifier(MetaMetricBaseClassifier):
    """Implementation of `ProtoNet  <https://arxiv.org/abs/1703.05175>`_."""

    def __init__(self, head=dict(type='PrototypicalHead'), *args, **kwargs):
        self.head_cfg = copy.deepcopy(head)
        super(ProtoNetClassifier, self).__init__(head=head, *args, **kwargs)
