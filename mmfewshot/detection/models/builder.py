# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.builder import DETECTORS


def build_detector(cfg, train_cfg=None, test_cfg=None, logger=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    # get fixed parameters
    frozen_parameters = cfg.pop('frozen_parameters', None)

    model = DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
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
