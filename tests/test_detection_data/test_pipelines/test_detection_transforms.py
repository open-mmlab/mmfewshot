# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np

from mmfewshot.detection.datasets.pipelines.transforms import PIPELINES


def test_crop_resize_instance():
    transform = dict(
        type='CropResizeInstance',
        num_context_pixels=16,
        target_size=(320, 320))
    transform = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']
    results['gt_bboxes'] = np.array([[10, 10, 60, 60]])
    results = transform(results)
    assert results['img'].shape == (320, 320, 3)
    assert results['img_shape'] == (320, 320, 3)
    results['gt_bboxes'] = np.array([[10, 10, 50, 60]])
    results = transform(results)
    assert results['img'].shape == (320, 320, 3)
    assert results['img_shape'] == (320, 320, 3)
    repr_str = transform.__repr__
    assert repr_str


def test_generate_mask():
    transform = dict(type='GenerateMask', target_size=(224, 224))
    transform = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']
    results['gt_bboxes'] = np.array([[10, 10, 60, 60]])
    results = transform(results)
    assert results['img_shape'] == (224, 224, 4)
    assert results['img'].shape == (224, 224, 4)
    repr_str = transform.__repr__
    assert repr_str


def test_crop_instance():
    transform = dict(type='CropInstance', context_ratio=1 / 7.)
    transform = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']
    results['gt_bboxes'] = np.array([[10, 10, 80, 80]])
    results = transform(results)
    assert results['img_shape'] == (80, 80, 3)
    assert results['img'].shape == (80, 80, 3)
    repr_str = transform.__repr__
    assert repr_str


def test_resize_to_multi_scale():
    transform = dict(
        type='ResizeToMultiScale', multi_scales=[(32, 32), (64, 64)])
    resize_module = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    results = resize_module(results)
    assert len(results) == 2
    assert results[0]['img_shape'] == (18, 32, 3)
    assert results[1]['img_shape'] == (36, 64, 3)
    repr_str = transform.__repr__
    assert repr_str


def test_multi_image_random_crop():
    transform = dict(
        type='MultiImageRandomCrop', multi_crop_sizes=[(32, 32), (64, 64)])
    crop_module = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']

    results = crop_module([copy.deepcopy(results), copy.deepcopy(results)])
    assert len(results) == 2
    repr_str = transform.__repr__
    assert repr_str


def test_multi_image_random_flip():
    transform = dict(type='MultiImageRandomFlip', flip_ratio=1.)
    flip_module = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']

    results = flip_module([copy.deepcopy(results), copy.deepcopy(results)])
    assert len(results) == 2


def test_multi_image_normalize():
    transform = dict(
        type='MultiImageNormalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False)
    norm_module = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']

    results = norm_module([copy.deepcopy(results), copy.deepcopy(results)])
    assert len(results) == 2


def test_multi_image_pad():
    transform = dict(type='MultiImagePad', size_divisor=32)
    pad_module = PIPELINES.build(transform)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']

    results = pad_module([copy.deepcopy(results), copy.deepcopy(results)])
    assert len(results) == 2
