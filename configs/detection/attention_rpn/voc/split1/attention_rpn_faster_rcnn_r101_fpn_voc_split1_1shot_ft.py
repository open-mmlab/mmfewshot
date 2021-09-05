_base_ = [
    '../../../_base_/datasets/query_aware/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../attention_rpn_faster_rcnn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
num_support_ways = 2
num_support_shots = 1
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        repeat_times=500,
        dataset=dict(
            num_novel_shots=1,
            classes='NOVEL_CLASSES_SPLIT1',
        )),
    val=dict(classes='NOVEL_CLASSES_SPLIT1'),
    test=dict(classes='NOVEL_CLASSES_SPLIT1'),
    model_init=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_1SHOT')],
        num_novel_shots=1,
        classes='NOVEL_CLASSES_SPLIT1'))
evaluation = dict(interval=100, class_splits=['NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=100)
optimizer = dict(lr=0.001, momentum=0.9)
lr_config = dict(warmup=None, step=[100])
log_config = dict(interval=10)
runner = dict(max_iters=100)
# load_from = 'path of base training model'
load_from = 'work_dirs/' \
    'attention_rpn_faster_rcnn_r101_fpn_voc_split1_base_training/' \
    'latest.pth'
model = dict(
    frozen_parameters=['backbone'],
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
)
