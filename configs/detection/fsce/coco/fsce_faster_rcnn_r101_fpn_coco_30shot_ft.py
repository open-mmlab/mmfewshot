_base_ = [
    '../../_base_/datasets/finetune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../fsce_faster_rcnn_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        type='FewShotCocoDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='30SHOT')],
        num_novel_shots=30,
        num_base_shots=30))
evaluation = dict(interval=5000)
checkpoint_config = dict(interval=5000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.3, step=[30000])
runner = dict(max_iters=40000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'fsce_faster_rcnn_r101_fpn_coco_base_training/' \
    'model_reset_randinit.pth'
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5))))
