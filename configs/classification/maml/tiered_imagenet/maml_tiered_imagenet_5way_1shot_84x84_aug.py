_base_ = [
    '../../_base_/meta_test/tiered_imagenet_meta_test_5way_1shot_84x84.py',
    '../../_base_/runtime/iter_based_runtime.py',
    '../../_base_/schedules/adam_100000iter.py'
]

img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromBytes'),
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type='EpisodicDataset',
        num_episodes=100000,
        num_ways=5,
        num_shots=5,
        num_queries=16,
        dataset=dict(
            type='TieredImageNetDataset',
            data_prefix='data/tiered_imagenet',
            subset='train',
            pipeline=train_pipeline)),
    val=dict(
        meta_test_cfg=dict(
            fast_test=False, support=dict(batch_size=5, num_inner_steps=5))),
    test=dict(
        meta_test_cfg=dict(
            fast_test=False, support=dict(batch_size=5, num_inner_steps=5))))

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=8, grad_clip=None)
optimizer = dict(type='Adam', lr=0.001)
pin_memory = True
