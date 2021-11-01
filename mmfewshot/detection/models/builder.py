# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv.utils import ConfigDict, print_log
from mmdet.models.builder import DETECTORS


def build_detector(cfg: ConfigDict, logger: Optional[object] = None):
    """Build detector."""
    # get the prefix of fixed parameters
    frozen_parameters = cfg.pop('frozen_parameters', None)

    model = DETECTORS.build(cfg)
    model.init_weights()
    # freeze parameters by prefix
    if frozen_parameters is not None:
        print_log(f'Frozen parameters: {frozen_parameters}', logger)
        for name, param in model.named_parameters():
            for frozen_prefix in frozen_parameters:
                if frozen_prefix in name:
                    param.requires_grad = False
            if param.requires_grad:
                print_log(f'Training parameters: {name}', logger)
    return model
