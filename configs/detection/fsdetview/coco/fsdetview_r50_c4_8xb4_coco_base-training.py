_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco.py',
    '../../_base_/schedules/schedule.py', '../fsdetview_r50_c4.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=1000, step=[80000])
runner = dict(max_iters=80000)
optimizer = dict(lr=0.01)
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60)))
checkpoint_config = dict(interval=10000)
evaluation = dict(interval=10000, metric='bbox', classwise=True)
