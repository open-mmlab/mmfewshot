_base_ = [
    '../../../_base_/datasets/query_aware/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../attention-rpn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
num_support_ways = 2
num_support_shots = 1
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='Attention_RPN', setting='SPLIT3_1SHOT')],
            num_novel_shots=1,
            num_base_shots=1,
            min_bbox_area=0,
            classes='ALL_CLASSES_SPLIT3',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'),
    model_init=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=100, class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=100)
optimizer = dict(lr=0.001, momentum=0.9)
lr_config = dict(warmup=None, step=[300])
log_config = dict(interval=10)
runner = dict(max_iters=300)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/attention_rpn_r50_c4_voc-split3_base-training/latest.pth'
model = dict(
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ],
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
)
