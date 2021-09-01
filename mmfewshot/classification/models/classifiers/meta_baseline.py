from mmcls.models.builder import CLASSIFIERS

from .meta_metric_base import MetaMetricBaseClassifier


@CLASSIFIERS.register_module()
class MetaBaselineClassifier(MetaMetricBaseClassifier):
    """Implementation of `MetaBaseline <https://arxiv.org/abs/2003.04390>`_.

    Args:
        head (dict): Config of classification head for training.
        meta_test_head (dict): Config of classification head for meta testing.
            the `meta_test_head` only will be built and run in meta testing.
    """

    def __init__(self, head=dict(type='MetaBaselineHead'), *args, **kwargs):
        super(MetaBaselineClassifier, self).__init__(
            head=head, *args, **kwargs)
