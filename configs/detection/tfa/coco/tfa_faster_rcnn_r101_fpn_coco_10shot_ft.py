_base_ = [
    '../../_base_/datasets/finetune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../tfa_faster_rcnn_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        type='FewShotCocoDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='10SHOT')],
        num_novel_shots=10,
        num_base_shots=10))
evaluation = dict(interval=80000)
checkpoint_config = dict(interval=80000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[144000])
runner = dict(max_iters=160000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'tfa_faster_rcnn_r101_fpn_coco_base_training/' \
    'model_reset_randinit.pth'
model = dict(roi_head=dict(bbox_head=dict(num_classes=80)))
