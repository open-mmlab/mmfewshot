_base_ = [
    '../../../_base_/datasets/finetune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='TFA', setting='SPLIT2_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT2',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=12000,
    class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=12000)
optimizer = dict(lr=0.005)
lr_config = dict(
    warmup_iters=10, step=[
        11000,
    ])
runner = dict(max_iters=12000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'tfa_faster_rcnn_r101_fpn_voc_split2_base_training/' \
    'model_reset_surgery.pth'
model = dict(
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            scale=20)))
