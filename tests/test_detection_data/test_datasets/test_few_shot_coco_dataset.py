# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile

import numpy as np
from mmdet.apis.train import set_random_seed

from mmfewshot.detection.datasets.coco import (FewShotCocoCopyDataset,
                                               FewShotCocoDataset)


def test_few_shot_coco_dataset():
    set_random_seed(0)
    # test regular annotation loading
    data_config = {
        'ann_cfg': [{
            'type': 'ann_file',
            'ann_file': 'tests/data/coco_sample.json'
        }],
        'img_prefix':
        '',
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'car')
    }
    dataset = FewShotCocoDataset(**data_config)

    assert len(dataset.data_infos) == 2
    assert dataset.CLASSES == ('bus', 'car')
    count = 0
    for datainfo in dataset.data_infos:
        count += len(datainfo['ann']['labels'])
    assert count == 3
    # test regular annotation loading
    data_config = {
        'ann_cfg': [{
            'type':
            'ann_file',
            'ann_file':
            f'tests/data/few_shot_coco_split/{class_name}.json'
        } for class_name in ['bus', 'car', 'cat', 'dog', 'person']],
        'img_prefix':
        '',
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'car', 'cat', 'dog')
    }
    dataset = FewShotCocoDataset(**data_config)
    assert len(dataset.data_infos) == 26
    count = 0
    for datainfo in dataset.data_infos:
        count += len(datainfo['ann']['labels'])
    assert count == 40

    data_config = {
        'ann_cfg': [{
            'type':
            'ann_file',
            'ann_file':
            f'tests/data/few_shot_coco_split/{class_name}.json'
        } for class_name in ['bus']],
        'img_prefix':
        '',
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'car', 'cat', 'dog')
    }
    dataset = FewShotCocoDataset(**data_config)
    dataset.SPLIT['test_a'] = ('bus', 'car')
    dataset.SPLIT['test_b'] = ('cat', 'dog')
    result = dataset.evaluate(
        [[np.array([[10, 10, 100, 100, 0.8]])] * 4] * 6,
        class_splits=['test_a', 'test_b'])
    for split in ['', 'test_a ', 'test_b ']:
        for metric in ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75']:
            assert split + metric in result

    # test default class split
    for class_split in ['ALL_CLASSES', 'NOVEL_CLASSES', 'BASE_CLASSES']:
        data_config['classes'] = class_split
        FewShotCocoDataset(**data_config)

    # test loading annotation with specific class
    data_config = {
        'ann_cfg': [{
            'type': 'ann_file',
            'ann_file': 'tests/data/coco_sample.json'
        }, {
            'type': 'ann_file',
            'ann_file': 'tests/data/coco_sample.json'
        }],
        'img_prefix':
        '',
        'ann_shot_filter': {
            'bus': 1,
            'car': 0
        },
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'car'),
    }
    dataset = FewShotCocoDataset(**data_config)
    count = 0
    for datainfo in dataset.data_infos:
        count += len(datainfo['ann']['labels'])
        for i in range(len(datainfo['ann']['labels'])):
            assert datainfo['ann']['labels'][i] == 0
    assert count == 1

    # test copy dataset
    data_config['ann_cfg'] = [{
        'data_infos': copy.deepcopy(dataset.data_infos)
    }]
    data_config['ann_shot_filter'] = None
    copy_dataset = FewShotCocoCopyDataset(**data_config)
    count = 0
    for data_info in copy_dataset.data_infos:
        count += len(data_info['ann']['bboxes'])
    assert count == 1

    # test save and load dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_data_infos(tmpdir + f'{os.sep}ann.json')
        data_config['ann_cfg'] = [{
            'type': 'saved_dataset',
            'ann_file': tmpdir + f'{os.sep}ann.json'
        }]
        dataset = FewShotCocoDataset(**data_config)
        count = 0
        for data_info in dataset.data_infos:
            count += len(data_info['ann']['bboxes'])
        assert count == 1

        # test ann shot filter
        data_config = {
            'ann_cfg': [{
                'type':
                'ann_file',
                'ann_file':
                f'tests/data/few_shot_coco_split/{class_name}.json'
            } for class_name in ['bus']],
            'img_prefix':
            '',
            'multi_pipelines': {
                'query': [{
                    'type': 'LoadImageFromFile'
                }],
                'support': [{
                    'type': 'LoadImageFromFile'
                }]
            },
            'classes':
            'ALL_CLASSES',
            'num_novel_shots':
            2,
            'num_base_shots':
            2
        }
        dataset = FewShotCocoDataset(**data_config)
        count = 0
        for datainfo in dataset.data_infos:
            count += len(datainfo['ann']['labels'])
        assert count == 2
