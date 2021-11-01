_base_ = [
    '../../_base_/datasets/two_branch/base_coco.py',
    '../../_base_/schedules/schedule.py', '../mpsr_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
evaluation = dict(interval=20000)
checkpoint_config = dict(interval=10000)
optimizer = dict(lr=0.005)
lr_config = dict(warmup_iters=500, warmup_ratio=1. / 3, step=[56000, 70000])
runner = dict(max_iters=80000)
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=60)))
