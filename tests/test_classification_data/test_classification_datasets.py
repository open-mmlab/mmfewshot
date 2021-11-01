# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from mmfewshot.classification.datasets import (CUBDataset, MiniImageNetDataset,
                                               TieredImageNetDataset)


def test_cub_dataset():
    CUBDataset.load_annotations = MagicMock(return_value=[])
    dataset = CUBDataset(
        data_prefix='',
        subset='train',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 100
    dataset = CUBDataset(
        data_prefix='',
        subset='val',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 50
    dataset = CUBDataset(
        data_prefix='',
        subset='test',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 50


def test_mini_imagenet_dataset():
    MiniImageNetDataset.load_annotations = MagicMock(return_value=[])
    dataset = MiniImageNetDataset(
        data_prefix='',
        subset='train',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 64
    dataset = MiniImageNetDataset(
        data_prefix='',
        subset='val',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 16
    dataset = MiniImageNetDataset(
        data_prefix='',
        subset='test',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 20


def test_tiered_imagenet_dataset():
    TieredImageNetDataset.load_annotations = MagicMock(return_value=[])
    dataset = TieredImageNetDataset(
        data_prefix='',
        subset='train',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 351
    dataset = TieredImageNetDataset(
        data_prefix='',
        subset='val',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 97
    dataset = TieredImageNetDataset(
        data_prefix='',
        subset='test',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 160
