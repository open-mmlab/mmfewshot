# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import numpy as np
from mmdet.apis.train import set_random_seed

from mmfewshot.detection.datasets.builder import build_dataset


def test_nway_kshot_dataset():
    set_random_seed(0)
    # test regular and few shot annotations
    data_configs = [{
        'type': 'NWayKShotDataset',
        'num_support_ways': 3,
        'num_support_shots': 1,
        'num_used_support_shots': 1,
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
                'query': [{
                    'type': 'LoadImageFromFile'
                }],
                'support': [{
                    'type': 'LoadImageFromFile'
                }]
            },
            'classes': ('car', 'dog', 'chair'),
        }
    }, {
        'type': 'NWayKShotDataset',
        'num_support_ways': 3,
        'num_support_shots': 1,
        'num_used_support_shots': 1,
        'dataset': {
            'type':
            'FewShotCocoDataset',
            'ann_cfg': [{
                'type':
                'ann_file',
                'ann_file':
                f'tests/data/few_shot_coco_split/{class_name}.json'
            } for class_name in ['bus', 'car', 'cat', 'dog', 'person']],
            'img_prefix':
            'tests/data',
            'multi_pipelines': {
                'query': [{
                    'type': 'LoadImageFromFile'
                }],
                'support': [{
                    'type': 'LoadImageFromFile'
                }]
            },
            'classes': ('bus', 'car', 'cat')
        }
    }]
    for data_config in data_configs:
        # test query dataset with 5 way 1 shot
        nway_kshot_dataset = build_dataset(cfg=data_config)
        assert nway_kshot_dataset._mode == 'query'
        assert np.sum(nway_kshot_dataset.flag) == 0
        assert isinstance(nway_kshot_dataset[0], dict)
        # test support dataset with 5 way 1 shot
        nway_kshot_dataset.convert_query_to_support(support_dataset_len=5)
        batch_index = nway_kshot_dataset.batch_index
        assert nway_kshot_dataset._mode == 'support'
        assert nway_kshot_dataset.flag.shape[0] == 5
        assert len(batch_index) == 5
        assert len(batch_index[0]) == 3
        assert len(batch_index[0][0]) == 2
        for i in range(5):
            support_batch = nway_kshot_dataset[i]
            assert isinstance(support_batch, list)
            count_classes = [0 for _ in range(3)]
            for item in support_batch:
                count_classes[item['ann_info']['labels'][0]] += 1
            for count in count_classes:
                assert count <= 1
        # test save dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            nway_kshot_dataset.save_data_infos(tmpdir + 'ann.json')
