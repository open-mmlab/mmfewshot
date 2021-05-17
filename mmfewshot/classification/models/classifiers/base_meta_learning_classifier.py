from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers import BaseClassifier


# just an example
@CLASSIFIERS.register_module()
class BaseMetaLearingClassifier(BaseClassifier):
    pass
