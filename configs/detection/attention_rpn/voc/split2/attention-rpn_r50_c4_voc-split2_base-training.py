_base_ = [
    '../../../_base_/datasets/query_aware/base_voc.py',
    '../../../_base_/schedules/schedule.py', '../../attention-rpn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
num_support_ways = 2
num_support_shots = 10
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(classes='BASE_CLASSES_SPLIT2')),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'),
    model_init=dict(classes='BASE_CLASSES_SPLIT2'))
optimizer = dict(
    lr=0.004,
    momentum=0.9,
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=2.0)}))
lr_config = dict(warmup_iters=500, warmup_ratio=0.1, step=[16000])
runner = dict(max_iters=18000)
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)

model = dict(
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
)
