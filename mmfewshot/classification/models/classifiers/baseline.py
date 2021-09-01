from mmcls.models.builder import CLASSIFIERS

from .finetune_base import FinetuneBaseClassifier


@CLASSIFIERS.register_module()
class BaselineClassifier(FinetuneBaseClassifier):
    """Implementation of Baseline classifier.

    Args:
        head (dict): Config of classification head for training.
        meta_test_head (dict): Config of classification head for meta testing.
            the `meta_test_head` only will be built and run in meta testing.
    """

    def __init__(self,
                 head=dict(
                     type='LinearHead', num_classes=100, in_channels=1024),
                 meta_test_head=dict(
                     type='LinearHead', num_classes=5, in_channels=1024),
                 *args,
                 **kwargs):
        super(BaselineClassifier, self).__init__(
            head=head, meta_test_head=meta_test_head, *args, **kwargs)
