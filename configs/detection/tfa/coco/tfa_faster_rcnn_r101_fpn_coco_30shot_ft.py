_base_ = [
    '../../_base_/datasets/finetune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../tfa_faster_rcnn_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='TFA', setting='30SHOT')],
            num_novel_shots=30,
            num_base_shots=30)))
evaluation = dict(interval=120000)
checkpoint_config = dict(interval=120000)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup_iters=10, step=[
        216000,
    ])
runner = dict(max_iters=240000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'faster_rcnn_r101_fpn_coco_base_training/' \
    'model_reset_surgery.pth'
model = dict(roi_head=dict(bbox_head=dict(num_classes=80)))
