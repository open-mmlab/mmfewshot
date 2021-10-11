_base_ = [
    '../../_base_/meta_test/mini_imagenet_meta_test_5way_5shot_84x84.py',
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
    workers_per_gpu=6,
    train=dict(
        type='MiniImageNetDataset',
        data_prefix='data/mini_imagenet',
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
