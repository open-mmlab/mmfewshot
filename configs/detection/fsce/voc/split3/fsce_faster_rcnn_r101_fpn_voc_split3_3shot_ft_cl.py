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
            ann_cfg=[dict(method='FSCE', setting='SPLIT3_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT3')),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=700, class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=5000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.5, step=[4000, 6000])
runner = dict(max_iters=8000)
custom_hooks = [
    dict(
        type='ContrastiveLossDecayHook',
        decay_steps=(3000, 6000),
        decay_rate=0.5)
]
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'fsce_faster_rcnn_r101_fpn_voc_split3_base_training/' \
    'model_reset_surgery.pth'
model = dict(
    frozen_parameters=[
        'backbone',
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        type='ContrastiveRoIHead',
        bbox_head=dict(
            type='ContrastiveBBoxHead',
            num_shared_fcs=2,
            num_classes=20,
            mlp_head_channels=128,
            with_weight_decay=True,
            loss_contrast=dict(
                type='SupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.6,
                loss_weight=0.2,
                reweight_type='none'),
            scale=20,
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001)),
                dict(
                    type='Caffe2Xavier',
                    override=dict(
                        type='Caffe2Xavier', name='contrastive_head'))
            ])),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
            sampler=dict(num=256))))
