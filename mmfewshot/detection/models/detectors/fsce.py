# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class FSCE(TwoStageDetector):
    """Implementation of `FSCE <https://arxiv.org/abs/2103.05950>`_"""
