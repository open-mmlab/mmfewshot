# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.apis.train import set_random_seed

from mmfewshot.detection.datasets.builder import (build_dataloader,
                                                  build_dataset)


def test_two_branch_dataloader():
    set_random_seed(0)
    data_config = {
        'type': 'TwoBranchDataset',
        'dataset': {
            'type':
            'FewShotVOCDataset',
            'ann_cfg': [{
                'type': 'ann_file',
                'ann_file': 'tests/data/few_shot_voc_split/1.txt'
            }],
            'img_prefix':
            'tests/data/VOCdevkit/',
            'multi_pipelines': {
                'main': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                'auxiliary': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]
            },
            'classes':
            'BASE_CLASSES_SPLIT1',
        },
        'auxiliary_dataset': {
            'copy_from_main_dataset': True,
            'instance_wise': True,
            'dataset_name': 'auxiliary_dataset'
        }
    }
    two_branch_dataset = build_dataset(cfg=data_config)
    two_branch_dataloader = build_dataloader(
        two_branch_dataset,
        samples_per_gpu=2,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=True,
        data_cfg=dict(
            samples_per_gpu=2,
            workers_per_gpu=0,
            auxiliary_samples_per_gpu=1,
            auxiliary_workers_per_gpu=0),
        seed=0)
    assert len(two_branch_dataloader) == 5
    data_iter = iter(two_branch_dataloader)
    data_batch = next(data_iter)
    assert len(data_batch['main_data']['img_metas'].data[0]) == 2
    assert len(data_batch['auxiliary_data']['img_metas'].data[0]) == 1
    data_batch = next(data_iter)
    assert len(data_batch['main_data']['img_metas'].data[0]) == 2
    assert len(data_batch['auxiliary_data']['img_metas'].data[0]) == 1
