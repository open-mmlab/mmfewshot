_base_ = [
    '../../../_base_/datasets/two_branch/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../mpsr_faster_rcnn_r101_fpn.py', '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    save_dataset=True,
    train=dict(
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MPSR', setting='SPLIT1_1SHOT')],
            num_novel_shots=1,
            num_base_shots=1,
            classes='ALL_CLASSES_SPLIT1')),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=2000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=2000)
optimizer = dict(lr=0.005)
lr_config = dict(warmup_iters=500, warmup_ratio=1. / 3, step=[1300, 1700])
runner = dict(max_iters=2000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'mpsr_faster_rcnn_r101_fpn_voc_split1_base_training/' \
    'latest.pth'
