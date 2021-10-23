from mmdet.models.builder import DETECTORS

from .meta_rcnn import MetaRCNN


@DETECTORS.register_module()
class FSDetView(MetaRCNN):
    """Implementation of `FSDetView <https://arxiv.org/abs/1908.01998>`_."""
