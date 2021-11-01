# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

from mmfewshot.detection.datasets.pipelines.formatting import PIPELINES


def test_multi_image_format_bundle():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../data'),
        img_info=dict(filename='color.jpg'))
    load = dict(type='LoadImageFromFile')
    load = PIPELINES.build(load)
    bundle = dict(type='MultiImageFormatBundle')
    bundle = PIPELINES.build(bundle)
    results = load(results)
    assert 'pad_shape' not in results
    assert 'scale_factor' not in results
    assert 'img_norm_cfg' not in results
    results = bundle([copy.deepcopy(results), copy.deepcopy(results)])
    assert 'pad_shape' in results[0]
    assert 'scale_factor' in results[0]
    assert 'img_norm_cfg' in results[0]


def test_multi_image_collect():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../data'),
        img_info=dict(filename='color.jpg'))
    load = dict(type='LoadImageFromFile')
    load = PIPELINES.build(load)
    collect = dict(
        type='MultiImageCollect',
        keys=['img'],
        meta_keys=('filename', 'ori_filename'))
    collect = PIPELINES.build(collect)
    results = load(results)
    results = collect([results, results, results])
    assert 'img_scale_0' in results
    assert 'img_scale_1' in results
    assert 'img_scale_2' in results
