from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import BaseDetector, FasterRCNN


@DETECTORS.register_module()
class BaseMetaLearingDetector(BaseDetector):
    pass


@DETECTORS.register_module()
class TestDetection(FasterRCNN):
    pass
