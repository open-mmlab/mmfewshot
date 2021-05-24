import numpy as np

from mmfewshot.apis.train import set_random_seed
from mmfewshot.detection.datasets.builder import build_dataset


def test_query_aware_dataset():
    set_random_seed(2023)
    # test regular annotations
    dataconfig = {
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
                'query': [{
                    'type': 'LoadImageFromFile'
                }],
                'support': [{
                    'type': 'LoadImageFromFile'
                }]
            },
            'classes': ('dog', 'chair', 'car'),
            'merge_dataset':
            True
        }
    }
    # test query dataset with 5 way 2 shot
    query_aware_dataset = build_dataset(cfg=dataconfig)

    assert np.sum(query_aware_dataset.flag) == 0
    # print(query_aware_dataset.data_infos_by_class)
    # self.data_infos_by_class = {
    #  0: [(0, 0)],
    #  1: [(1, 0), (3, 0), (3, 1), (3, 2)],
    #  2: [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
    #  }
    assert query_aware_dataset.sample_support_shots(0, 0, True) == \
           [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    support = query_aware_dataset.sample_support_shots(0, 1, False)
    assert len(set(support)) == 4
    support = query_aware_dataset.sample_support_shots(1, 1, False)
    assert len(set(support)) == 3
    support = query_aware_dataset.sample_support_shots(3, 1, False)
    assert len(set(support)) == 1
    support = query_aware_dataset.sample_support_shots(3, 2)
    assert len(set(support)) == 5
    support = query_aware_dataset.sample_support_shots(3, 0)
    assert len(set(support)) == 1

    dataconfig = {
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
    }

    query_aware_dataset = build_dataset(cfg=dataconfig)

    assert np.sum(query_aware_dataset.flag) == 0
    # print(query_aware_dataset.data_infos_by_class)
    # self.data_infos_by_class = {
    # 0: [(0, 0)],
    # 1: [(1, 0)],
    # 2: [(2, 0), (2, 1)],
    # 3: [(3, 0), (3, 1)],
    # 4: [(4, 0), (5, 0)]}
    assert query_aware_dataset.sample_support_shots(0, 0, True) == \
           [(0, 0), (0, 0)]
    support = query_aware_dataset.sample_support_shots(0, 1, False)
    assert len(set(support)) == 1
    support = query_aware_dataset.sample_support_shots(3, 0)
    assert len(set(support)) == 1
    assert len(support) == 2
    support = query_aware_dataset.sample_support_shots(3, 2)
    assert len(set(support)) == 2

    batch = query_aware_dataset[0]
    assert len(batch['support_data']) == 6
    assert batch['query_data']['ann_info']['labels'][0] == \
           batch['support_data'][0]['ann_info']['labels'][0]
    assert batch['query_data']['ann_info']['labels'][0] == \
           batch['support_data'][1]['ann_info']['labels'][0]
    assert batch['support_data'][2]['ann_info']['labels'][0] == \
           batch['support_data'][3]['ann_info']['labels'][0]
    assert batch['support_data'][4]['ann_info']['labels'][0] == \
           batch['support_data'][5]['ann_info']['labels'][0]
