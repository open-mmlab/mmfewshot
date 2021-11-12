# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.datasets.builder import DATASETS, PIPELINES

from .base import BaseFewShotDataset
from .builder import (build_dataloader, build_dataset,
                      build_meta_test_dataloader)
from .cub import CUBDataset
from .dataset_wrappers import EpisodicDataset, MetaTestDataset
from .mini_imagenet import MiniImageNetDataset
from .pipelines import LoadImageFromBytes
from .tiered_imagenet import TieredImageNetDataset
from .utils import label_wrapper

__all__ = [
    'build_dataloader', 'build_dataset', 'DATASETS', 'PIPELINES', 'CUBDataset',
    'LoadImageFromBytes', 'build_meta_test_dataloader', 'MiniImageNetDataset',
    'TieredImageNetDataset', 'label_wrapper', 'BaseFewShotDataset',
    'EpisodicDataset', 'MetaTestDataset'
]
