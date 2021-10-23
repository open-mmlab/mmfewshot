import numpy as np

from mmfewshot.apis.train import set_random_seed
from mmfewshot.detection.datasets.builder import build_dataset


def test_nway_kshot_dataset():
    set_random_seed(2021)
    # test regular and few shot annotations
    dataconfigs = [{
        'type': 'NWayKShotDataset',
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
    }, {
        'type': 'NWayKShotDataset',
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
    }]
    for dataconfig in dataconfigs:
        # test query dataset with 5 way 1 shot
        nway_kshot_dataset = build_dataset(cfg=dataconfig)
        assert nway_kshot_dataset.data_type == 'query'
        assert np.sum(nway_kshot_dataset.flag) == 0
        assert isinstance(nway_kshot_dataset[0], dict)
        # test support dataset with 5 way 1 shot
        nway_kshot_dataset.convert_query_to_support(support_dataset_len=2)
        batch_index = nway_kshot_dataset.batch_index
        assert nway_kshot_dataset.data_type == 'support'
        assert nway_kshot_dataset.flag.shape[0] == 2
        assert len(batch_index) == 2
        assert len(batch_index[0]) == 5
        assert len(batch_index[0][0]) == 2
        # test batch of support dataset with 5 way 1 shot
        support_batch = nway_kshot_dataset[0]
        assert isinstance(support_batch, list)
        count_classes = [0 for _ in range(5)]
        for item in support_batch:
            count_classes[item['ann_info']['labels'][0]] += 1
        for count in count_classes:
            assert count == 1
        # test support dataset with 4 way 2 shot
        dataconfig['support_way'] = 4
        dataconfig['support_shot'] = 2
        nway_kshot_dataset = build_dataset(cfg=dataconfig)
        assert nway_kshot_dataset.data_type == 'query'
        assert np.sum(nway_kshot_dataset.flag) == 0
        assert isinstance(nway_kshot_dataset[0], dict)
        # test support dataset with 4 way 2 shot
        nway_kshot_dataset.convert_query_to_support(support_dataset_len=3)
        batch_index = nway_kshot_dataset.batch_index
        assert nway_kshot_dataset.data_type == 'support'
        assert nway_kshot_dataset.flag.shape[0] == 3
        assert len(batch_index) == 3
        assert len(batch_index[0]) == 4 * 2
        assert len(batch_index[0][0]) == 2
        for i in range(len(nway_kshot_dataset.CLASSES)):
            assert len(nway_kshot_dataset.data_infos_by_class[i]) >= 2
        # test batch of support dataset with 4 way 2 shot
        for idx in range(3):
            support_batch = nway_kshot_dataset[idx]
            assert isinstance(support_batch, list)
            count_classes = [0 for _ in range(5)]
            dog_ann = None
            for item in support_batch:
                label = item['ann_info']['labels'][0]
                count_classes[label] += 1
                # test whether dog label is repeat or not
                # (only one dog instance)
                if label == 1:
                    if dog_ann is None:
                        dog_ann = item['ann_info']['bboxes']
                    else:
                        assert (dog_ann == item['ann_info']['bboxes']).all()
            # test number of classes sampled
            # 4 class have 2 shots 1 class has 0 shot
            is_skip = False
            for count in count_classes:
                if count == 0:
                    assert not is_skip
                    is_skip = True
                else:
                    assert count == 2
