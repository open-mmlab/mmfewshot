import random

import numpy as np
import torch
from mmcls.apis.train import train_model as train_classifier
from mmdet.apis.train import train_detector


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(*args, task_type='mmdet', **kwargs):
    if task_type == 'mmdet':
        train_detector(*args, **kwargs)
    elif task_type == 'mmcls':
        train_classifier(*args, **kwargs)
    else:
        raise NotImplementedError
