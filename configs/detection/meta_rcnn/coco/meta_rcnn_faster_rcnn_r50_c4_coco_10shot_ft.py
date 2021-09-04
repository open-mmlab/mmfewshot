_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../meta_rcnn_faster_rcnn_r50_c4.py',
    '../../_base_/default_runtime.py'
]
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data = dict(
    train=dict(
        save_dataset=True,
        repeat_times=100,
        num_used_support_shots=10,
        dataset=dict(
            num_novel_shots=10,
            num_base_shots=10,
        )),
    model_init=dict(num_novel_shots=10, num_base_shots=10))
evaluation = dict(interval=1000)
checkpoint_config = dict(interval=1000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[1000])
runner = dict(max_iters=1000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'meta_rcnn_faster_rcnn_r50_c4_coco_base_training/' \
    'latest.pth'
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80, num_meta_classes=80), ))
