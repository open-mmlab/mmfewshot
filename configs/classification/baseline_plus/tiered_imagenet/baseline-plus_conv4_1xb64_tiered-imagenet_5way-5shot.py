_base_ = [
    '../../_base_/meta_test/tiered-imagenet_meta-test_5way-5shot.py',
    '../../_base_/runtime/epoch_based_runtime.py',
    '../../_base_/schedules/sgd_200epoch.py'
]

img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

meta_finetune_cfg = dict(
    num_steps=600,
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001))

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='TieredImageNetDataset',
        data_prefix='data/tiered_imagenet',
        subset='train',
        pipeline=train_pipeline),
    val=dict(
        meta_test_cfg=dict(
            support=dict(
                batch_size=4, drop_last=True, train=meta_finetune_cfg))),
    test=dict(
        meta_test_cfg=dict(
            support=dict(
                batch_size=4, drop_last=True, train=meta_finetune_cfg))))

model = dict(
    type='BaselinePlusClassifier',
    backbone=dict(type='Conv4'),
    head=dict(type='CosineDistanceHead', num_classes=351, in_channels=1600),
    meta_test_head=dict(
        type='CosineDistanceHead', num_classes=5, in_channels=1600))
