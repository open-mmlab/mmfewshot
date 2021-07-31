_base_ = [
    '../../_base_/datasets/finetune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py',
    '../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='FSCE', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10)))
evaluation = dict(interval=700)
checkpoint_config = dict(interval=5000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.3, step=[20000])
runner = dict(max_iters=30000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'faster_rcnn_r101_fpn_coco_base_training/' \
    'model_reset_surgery.pth'
model = dict(
    frozen_parameters=[
        'backbone',
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=80,
            scale=20)),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000), rcnn=dict(sampler=dict(num=256))))
