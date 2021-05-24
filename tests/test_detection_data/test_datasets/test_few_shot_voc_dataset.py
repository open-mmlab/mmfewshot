from mmfewshot.apis.train import set_random_seed
from mmfewshot.detection.datasets.voc import FewShotVOCDataset


def test_few_shot_voc_dataset():
    set_random_seed(2021)
    # test regular annotation loading
    dataconfig = {
        'ann_file': 'tests/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        'img_prefix': 'tests/data/VOCdevkit/',
        'pipeline': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('car', 'dog', 'chair')
    }
    few_shot_custom_dataset = FewShotVOCDataset(**dataconfig)

    # filter image without labels
    assert len(few_shot_custom_dataset.data_infos) == 4
    assert few_shot_custom_dataset.CLASSES == ('car', 'dog', 'chair')
    # test loading annotation with specific class
    dataconfig = {
        'ann_file': 'tests/data/few_shot_voc_split/1.txt',
        'img_prefix': 'tests/data/VOCdevkit/',
        'ann_shot_filter': {
            'aeroplane': 10
        },
        'pipeline': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('car', 'dog', 'chair', 'aeroplane'),
    }
    few_shot_custom_dataset = FewShotVOCDataset(**dataconfig)
    count = 0
    for datainfo in few_shot_custom_dataset.data_infos:
        count += len(datainfo['ann']['bboxes'])
    assert count == 5

    # test loading annotation with specific class with specific shot
    dataconfig = {
        'ann_file': 'tests/data/few_shot_voc_split/1.txt',
        'img_prefix': 'tests/data/VOCdevkit/',
        'ann_shot_filter': {
            'aeroplane': 2
        },
        'pipeline': {
            'query': [{
                'type': 'LoadImageFromFile'
            }],
            'support': [{
                'type': 'LoadImageFromFile'
            }]
        },
        'classes': ('car', 'dog', 'chair', 'aeroplane'),
    }
    few_shot_custom_dataset = FewShotVOCDataset(**dataconfig)
    count = 0
    for datainfo in few_shot_custom_dataset.data_infos:
        count += len(datainfo['ann']['bboxes'])
    assert count == 2
