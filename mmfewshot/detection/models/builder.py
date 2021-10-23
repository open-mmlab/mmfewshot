# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv.utils import ConfigDict
from mmdet.models.builder import DETECTORS


def build_detector(cfg: ConfigDict, logger: Optional[object] = None):
    """Build detector."""
    # get fixed parameters
    frozen_parameters = cfg.pop('frozen_parameters', None)

    model = DETECTORS.build(cfg)
    model.init_weights()
    # freeze parameters by prefix
    if frozen_parameters is not None:
        logger.info(f'Frozen parameters: {frozen_parameters}')
        for name, param in model.named_parameters():
            for frozen_prefix in frozen_parameters:
                if frozen_prefix in name:
                    param.requires_grad = False
            if param.requires_grad:
                logger.info(f'Training parameters: {name}')
    return model
