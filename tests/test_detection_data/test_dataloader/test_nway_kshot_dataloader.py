# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.apis import set_random_seed

from mmfewshot.detection.datasets.builder import (build_dataloader,
                                                  build_dataset)


def test_nway_kshot_dataloader():
    set_random_seed(0)

    # test voc dataset
    data_config = {
        'type': 'NWayKShotDataset',
        'num_support_ways': 2,
        'num_support_shots': 1,
        'num_used_support_shots': 2,
        'shuffle_support': True,
        'repeat_times': 2,
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
                'query': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                'support': [
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]
            },
            'classes': ('car', 'dog', 'chair')
        }
    }
    nway_kshot_dataset = build_dataset(data_config)
    nway_kshot_dataloader = build_dataloader(
        nway_kshot_dataset,
        samples_per_gpu=2,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=2021)
    batch_index_a = \
        nway_kshot_dataloader.support_data_loader.dataset.batch_index
    for i, data_batch in enumerate(nway_kshot_dataloader):
        assert len(data_batch['query_data']['img_metas'].data[0]) == 2
        support_labels = data_batch['support_data']['gt_labels'].data[0]
        assert len(set(torch.cat(
            support_labels).tolist())) == data_config['num_support_ways']
        assert len(torch.cat(support_labels).tolist()) == \
               data_config['num_support_ways'] * \
               data_config['num_support_shots']
    batch_index_b = \
        nway_kshot_dataloader.support_data_loader.dataset.batch_index
    for i, data_batch in enumerate(nway_kshot_dataloader):
        assert len(data_batch['query_data']['img_metas'].data[0]) == 2
        support_labels = data_batch['support_data']['gt_labels'].data[0]
        assert len(set(torch.cat(
            support_labels).tolist())) == data_config['num_support_ways']
        assert len(torch.cat(support_labels).tolist()) == \
               data_config['num_support_ways'] * \
               data_config['num_support_shots']
    assert batch_index_a != batch_index_b
