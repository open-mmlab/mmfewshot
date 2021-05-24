import numpy as np

from mmfewshot.apis.train import set_random_seed
from mmfewshot.detection.datasets.builder import build_dataset


def test_merge_dataset():
    set_random_seed(2023)
    # test merge dataset load regular annotation
    dataconfig = {
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
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('person', 'dog', 'chair', 'car', 'aeroplane'),
        'merge_dataset':
        True
    }
    merge_dataset = build_dataset(cfg=dataconfig)
    count = [0 for _ in range(5)]
    for data_info in merge_dataset.dataset.data_infos:
        # test label merge
        if data_info['id'] == '000001':
            assert (np.sort(data_info['ann']['labels']) == np.array([0, 1
                                                                     ])).all()
        for label in data_info['ann']['labels']:
            count[label] += 1
    assert count == [4, 1, 4, 7, 5]

    # test merge dataset load annotation by class
    dataconfig = {
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
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('person', 'dog', 'chair', 'car', 'aeroplane'),
        'merge_dataset':
        True
    }
    merge_dataset = build_dataset(cfg=dataconfig)
    count = [0 for _ in range(5)]
    for data_info in merge_dataset.dataset.data_infos:
        # test label merge
        if data_info['id'] == '000001':
            assert (np.sort(data_info['ann']['labels']) == np.array([0, 1
                                                                     ])).all()
        for label in data_info['ann']['labels']:
            count[label] += 1
    assert count == [2, 1, 3, 3, 3]

    # test loading annotation with specific class with specific shot
    dataconfig = {
        'type':
        'FewShotCocoDataset',
        'ann_file': [
            'tests/data/few_shot_coco_split/bus.json',
            'tests/data/few_shot_coco_split/car.json',
            'tests/data/few_shot_coco_split/cat.json',
            'tests/data/few_shot_coco_split/dog.json',
            'tests/data/few_shot_coco_split/person.json',
        ],
        'img_prefix': ['', '', '', '', ''],
        'ann_shot_filter': [{
            'bus': 2
        }, {
            'car': 2
        }, {
            'cat': 3
        }, {
            'dog': 3
        }, {
            'person': 3
        }],
        'pipeline': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'car', 'cat', 'dog', 'person'),
        'merge_dataset':
        True
    }
    merge_dataset = build_dataset(cfg=dataconfig)
    count = [0 for _ in range(5)]
    for data_info in merge_dataset.dataset.data_infos:
        # test label merge
        for label in data_info['ann']['labels']:
            count[label] += 1
    assert count == [2, 2, 3, 3, 3]
