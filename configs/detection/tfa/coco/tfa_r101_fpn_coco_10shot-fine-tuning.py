_base_ = [
    '../../_base_/datasets/fine_tune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../tfa_r101_fpn.py',
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
model = dict(roi_head=dict(bbox_head=dict(num_classes=80)))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('work_dirs/tfa_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head.pth')
