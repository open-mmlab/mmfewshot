# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class TFA(TwoStageDetector):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""
