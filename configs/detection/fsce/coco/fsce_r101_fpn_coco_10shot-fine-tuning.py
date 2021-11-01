_base_ = [
    '../../_base_/datasets/fine_tune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../fsce_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        type='FewShotCocoDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='10SHOT')],
        num_novel_shots=10,
        num_base_shots=10))
evaluation = dict(interval=5000)
checkpoint_config = dict(interval=5000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.3, step=[20000])
runner = dict(max_iters=30000)
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5))))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
# load_from = 'path of base training model'
load_from = ('work_dirs/fsce_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head.pth')
