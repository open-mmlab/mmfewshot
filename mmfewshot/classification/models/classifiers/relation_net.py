import copy

from mmcls.models.builder import CLASSIFIERS

from .meta_metric_base import MetaMetricBaseClassifier


@CLASSIFIERS.register_module()
class RelationNetClassifier(MetaMetricBaseClassifier):
    """Implementation of `RelationNet  <https://arxiv.org/abs/1711.06025>`_."""

    def __init__(self,
                 head=dict(type='RelationHead', in_channels=64),
                 *args,
                 **kwargs):
        self.head_cfg = copy.deepcopy(head)
        super(RelationNetClassifier, self).__init__(head=head, *args, **kwargs)
