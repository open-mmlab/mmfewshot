_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco.py',
    '../../_base_/schedules/schedule.py', '../meta_rcnn_faster_rcnn_r50_c4.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup=None, step=[60000])
checkpoint_config = dict(interval=5000)
runner = dict(max_iters=90000)
optimizer = dict(lr=0.004)
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60), ))
