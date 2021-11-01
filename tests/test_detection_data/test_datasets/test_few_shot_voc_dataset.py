# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile

import numpy as np
from mmdet.apis import set_random_seed

from mmfewshot.detection.datasets.voc import (FewShotVOCCopyDataset,
                                              FewShotVOCDataset)


def test_few_shot_voc_dataset():
    set_random_seed(0)
    # test regular annotation loading
    data_config = {
        'ann_cfg': [{
            'type':
            'ann_file',
            'ann_file':
            'tests/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
        }],
        'img_prefix':
        'tests/data/VOCdevkit/',
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('car', 'dog', 'chair')
    }
    dataset = FewShotVOCDataset(**data_config)

    # filter image without labels
    assert len(dataset.data_infos) == 4
    assert dataset.CLASSES == ('car', 'dog', 'chair')

    # test loading annotation with specific class
    data_config = {
        'ann_cfg': [{
            'type': 'ann_file',
            'ann_file': 'tests/data/few_shot_voc_split/1.txt',
            'ann_classes': ['aeroplane']
        }],
        'img_prefix':
        'tests/data/VOCdevkit/',
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes':
        'BASE_CLASSES_SPLIT1',
    }
    dataset = FewShotVOCDataset(**data_config)
    assert len(dataset.CLASSES) == 15
    count = 0
    for data_info in dataset.data_infos:
        count += len(data_info['ann']['bboxes'])
    assert count == 5

    # test loading annotation with specific class with specific shot
    data_config = {
        'ann_cfg': [{
            'type': 'ann_file',
            'ann_file': 'tests/data/few_shot_voc_split/1.txt'
        }],
        'img_prefix':
        'tests/data/VOCdevkit/',
        'ann_shot_filter': {
            'aeroplane': 2
        },
        'multi_pipelines': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('aeroplane', ),
    }
    dataset = FewShotVOCDataset(**data_config)
    count = 0
    for data_info in dataset.data_infos:
        count += len(data_info['ann']['bboxes'])
    assert count == 2

    # test copy dataset
    data_config['ann_cfg'] = [{
        'data_infos': copy.deepcopy(dataset.data_infos)
    }]
    data_config['ann_shot_filter'] = None
    copy_dataset = FewShotVOCCopyDataset(**data_config)
    count = 0
    for data_info in copy_dataset.data_infos:
        count += len(data_info['ann']['bboxes'])
    assert count == 2

    # test save and load dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_data_infos(tmpdir + 'ann.json')
        data_config['ann_cfg'] = [{
            'type': 'saved_dataset',
            'ann_file': tmpdir + 'ann.json'
        }]
        dataset = FewShotVOCDataset(**data_config)
        count = 0
        for data_info in dataset.data_infos:
            count += len(data_info['ann']['bboxes'])
        assert count == 2
    dataset.SPLIT['test'] = ('aeroplane', )
    result = dataset.evaluate(
        [[np.array([[10, 10, 100, 100]])], [np.array([[10, 10, 100, 100]])]],
        class_splits=['test'])
    assert result['AP50'] == 0.0
    assert result['test: AP50'] == 0.0
    assert result['mAP'] == 0.0
