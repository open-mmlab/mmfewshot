_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco.py',
    '../../_base_/schedules/schedule.py', '../fsdetview_r50_c4.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=500, step=[110000], gamma=0.5)
checkpoint_config = dict(interval=10000)
runner = dict(max_iters=120000)
data = dict(train=dict(num_used_support_shots=500))
optimizer = dict(lr=0.005)
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60)))
