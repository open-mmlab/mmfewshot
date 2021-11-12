# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import pytest

from mmfewshot.classification.datasets import (CUBDataset, MiniImageNetDataset,
                                               TieredImageNetDataset)


def test_cub_dataset():
    CUBDataset.load_annotations = MagicMock(return_value=[])
    train_dataset = CUBDataset(
        data_prefix='',
        subset='train',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(train_dataset.CLASSES) == 100
    val_dataset = CUBDataset(
        data_prefix='',
        subset='val',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(val_dataset.CLASSES) == 50
    test_dataset = CUBDataset(
        data_prefix='',
        subset='test',
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(test_dataset.CLASSES) == 50
    dataset = CUBDataset(
        data_prefix='',
        subset=['test'],
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(dataset.CLASSES) == 50
    with pytest.raises(AssertionError):
        dataset = CUBDataset(
            data_prefix='',
            subset='none',
            pipeline=[dict(type='LoadImageFromFile')])
    random_train_dataset1 = CUBDataset(
        data_prefix='',
        subset='train',
        classes_id_seed=2,
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(random_train_dataset1.CLASSES) == 100
    random_train_dataset2 = CUBDataset(
        data_prefix='',
        subset='train',
        classes_id_seed=2,
        pipeline=[dict(type='LoadImageFromFile')])
    assert len(random_train_dataset2.CLASSES) == 100
    assert random_train_dataset1.CLASSES[0] == \
           random_train_dataset2.CLASSES[0]
    assert random_train_dataset1.CLASSES != train_dataset.CLASSES


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
