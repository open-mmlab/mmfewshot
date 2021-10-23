_base_ = [
    '../../_base_/datasets/query_aware/base_coco.py',
    '../../_base_/schedules/schedule.py',
    '../attention_rpn_faster_rcnn_r50_c4.py', '../../_base_/default_runtime.py'
]
num_support_ways = 2
num_support_shots = 10
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_train2017.json')
        ])),
    val=dict(ann_cfg=[
        dict(
            type='ann_file',
            ann_file='data/coco/annotations/instances_val2017.json')
    ]),
    test=dict(ann_cfg=[
        dict(
            type='ann_file',
            ann_file='data/coco/annotations/instances_val2017.json')
    ]),
    model_init=dict(ann_cfg=[
        dict(
            type='ann_file',
            ann_file='data/coco/annotations/instances_train2017.json')
    ]))

optimizer = dict(
    lr=0.004,
    momentum=0.9,
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=2.0)}))
lr_config = dict(warmup_iters=1000, warmup_ratio=0.1, step=[112000, 120000])
runner = dict(max_iters=120000)
evaluation = dict(interval=80000)
checkpoint_config = dict(interval=20000)
log_config = dict(interval=10)

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
