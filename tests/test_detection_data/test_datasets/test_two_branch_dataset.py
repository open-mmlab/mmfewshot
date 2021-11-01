# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

from mmdet.apis import set_random_seed

from mmfewshot.detection.datasets.builder import build_dataset


def test_two_branch_dataset():
    set_random_seed(0)
    # test regular annotations
    data_config = {
        'type': 'TwoBranchDataset',
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
                'main': [{
                    'type': 'LoadImageFromFile'
                }],
                'auxiliary': [{
                    'type': 'LoadImageFromFile'
                }]
            },
            'classes':
            'BASE_CLASSES_SPLIT1',
        },
        'auxiliary_dataset': {
            'copy_from_main_dataset': True,
            'instance_wise': True,
            'dataset_name': 'auxiliary_dataset'
        }
    }
    two_branch_dataset = build_dataset(cfg=data_config)
    assert two_branch_dataset._mode == 'main'
    assert len(two_branch_dataset) == 10
    two_branch_dataset.convert_main_to_auxiliary()
    assert two_branch_dataset._mode == 'auxiliary'
    assert len(two_branch_dataset) == 25
    # test save dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        two_branch_dataset.save_data_infos(tmpdir + 'ann.json')
