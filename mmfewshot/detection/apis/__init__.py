# Copyright (c) OpenMMLab. All rights reserved.
from .test import (multi_gpu_model_init, multi_gpu_test, single_gpu_model_init,
                   single_gpu_test)
from .train import train_detector

__all__ = [
    'train_detector', 'single_gpu_model_init', 'multi_gpu_model_init',
    'single_gpu_test', 'multi_gpu_test'
]
