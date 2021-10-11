import torch

from mmfewshot.detection.apis.train import set_random_seed
from mmfewshot.detection.datasets.builder import (build_dataloader,
                                                  build_dataset)


def test_dataloader():
    set_random_seed(2021)

    # test regular and few shot annotations
    data_configs = [{
        'type': 'NwayKshotDataset',
        'support_way': 5,
        'support_shot': 1,
        'dataset': {
            'type':
            'FewShotVOCDataset',
            'ann_file': [
                'tests/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                'tests/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
            ],
            'img_prefix': [
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
            ],
            'pipeline': {
                'query': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                'support': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]
            },
            'classes': ('person', 'dog', 'chair', 'car', 'aeroplane', 'train'),
            'merge_dataset':
            True
        }
    }, {
        'type': 'NwayKshotDataset',
        'support_way': 5,
        'support_shot': 1,
        'dataset': {
            'type':
            'FewShotVOCDataset',
            'ann_file': [
                'tests/data/few_shot_voc_split/1.txt',
                'tests/data/few_shot_voc_split/2.txt',
                'tests/data/few_shot_voc_split/3.txt',
                'tests/data/few_shot_voc_split/4.txt',
                'tests/data/few_shot_voc_split/5.txt'
            ],
            'img_prefix': [
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
            ],
            'ann_shot_filter': [{
                'person': 2
            }, {
                'dog': 2
            }, {
                'chair': 3
            }, {
                'car': 3
            }, {
                'aeroplane': 3
            }],
            'pipeline': {
                'query': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                'support': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]
            },
            'classes': ('person', 'dog', 'chair', 'car', 'aeroplane'),
            'merge_dataset':
            True
        }
    }]

    for data_config in data_configs:

        nway_kshot_dataset = build_dataset(cfg=data_config)
        nway_kshot_dataloader = build_dataloader(
            nway_kshot_dataset,
            samples_per_gpu=2,
            workers_per_gpu=0,
            num_gpus=1,
            dist=False,
            shuffle=True,
            seed=2021)

        for i, data_batch in enumerate(nway_kshot_dataloader):
            assert len(data_batch['query_data']['img_metas'].data[0]) == 2
            assert len(nway_kshot_dataloader.query_data_loader) == \
                   len(nway_kshot_dataloader.support_data_loader)
            support_labels = data_batch['support_data']['gt_labels'].data[0]
            assert len(set(torch.cat(
                support_labels).tolist())) == data_config['support_way']
            assert len(torch.cat(support_labels).tolist()) == \
                   data_config['support_way'] * data_config['support_shot']

    data_configs = [{
        'type': 'QueryAwareDataset',
        'support_way': 3,
        'support_shot': 5,
        'dataset': {
            'type':
            'FewShotVOCDataset',
            'ann_file': [
                'tests/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                'tests/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
            ],
            'img_prefix': [
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
            ],
            'pipeline': {
                'query': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                'support': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]
            },
            'classes': ('dog', 'chair', 'car'),
            'merge_dataset':
            True
        }
    }, {
        'type': 'QueryAwareDataset',
        'support_way': 3,
        'support_shot': 2,
        'dataset': {
            'type':
            'FewShotVOCDataset',
            'ann_file': [
                'tests/data/few_shot_voc_split/1.txt',
                'tests/data/few_shot_voc_split/2.txt',
                'tests/data/few_shot_voc_split/3.txt',
                'tests/data/few_shot_voc_split/4.txt',
                'tests/data/few_shot_voc_split/5.txt'
            ],
            'img_prefix': [
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
                'tests/data/VOCdevkit/',
            ],
            'ann_shot_filter': [{
                'person': 1
            }, {
                'dog': 1
            }, {
                'chair': 2
            }, {
                'car': 2
            }, {
                'aeroplane': 2
            }],
            'pipeline': {
                'query': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                'support': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]
            },
            'classes': ('person', 'dog', 'chair', 'car', 'aeroplane'),
            'merge_dataset':
            True
        }
    }]

    for data_config in data_configs:
        query_aware_dataset = build_dataset(cfg=data_config)
        query_aware_dataloader = build_dataloader(
            query_aware_dataset,
            samples_per_gpu=2,
            workers_per_gpu=0,
            num_gpus=1,
            dist=False,
            shuffle=True,
            seed=2021)

        for i, data_batch in enumerate(query_aware_dataloader):
            assert len(data_batch['query_data']['img_metas'].data[0]) == 2
            assert len(data_batch['query_data']['query_class'].tolist()) == 2
            support_labels = data_batch['support_data']['gt_labels'].data[0]
            half_batch = len(support_labels) // 2
            assert len(set(torch.cat(support_labels[:half_batch]).tolist())) \
                   == data_config['support_way']
            assert len(set(torch.cat(support_labels[half_batch:]).tolist())) \
                   == data_config['support_way']
