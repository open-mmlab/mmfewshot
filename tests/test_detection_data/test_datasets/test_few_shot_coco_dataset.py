from mmfewshot.apis.train import set_random_seed
from mmfewshot.detection.datasets.coco import FewShotCocoDataset


def test_few_shot_voc_dataset():
    set_random_seed(2021)
    # test regular annotation loading
    dataconfig = {
        'ann_file': 'tests/data/coco_sample.json',
        'img_prefix': '',
        'pipeline': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'car')
    }
    few_shot_custom_dataset = FewShotCocoDataset(**dataconfig)

    # filter image without labels
    assert len(few_shot_custom_dataset.data_infos) == 2
    assert few_shot_custom_dataset.CLASSES == ('bus', 'car')
    # test loading annotation with specific class
    dataconfig = {
        'ann_file': 'tests/data/few_shot_coco_split/bus.json',
        'img_prefix': '',
        'ann_shot_filter': {
            'bus': 5
        },
        'pipeline': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('bus', 'dog', 'car'),
    }
    few_shot_custom_dataset = FewShotCocoDataset(**dataconfig)
    count = 0
    for datainfo in few_shot_custom_dataset.data_infos:
        count += len(datainfo['ann']['labels'])
        for i in range(len(datainfo['ann']['labels'])):
            assert datainfo['ann']['labels'][i] == 0
    assert count == 5
