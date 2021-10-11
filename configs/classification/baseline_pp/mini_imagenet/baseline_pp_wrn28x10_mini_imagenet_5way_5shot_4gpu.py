_base_ = ['baseline_pp_mini_imagenet_5way_5shot_84x84_aug.py']

data = dict(samples_per_gpu=32, workers_per_gpu=2)
model = dict(
    type='BaselinePPClassifier',
    backbone=dict(type='WRN28x10'),
    head=dict(type='CosineDistanceHead', num_classes=100, in_channels=640),
    meta_test_head=dict(
        type='CosineDistanceHead', num_classes=5, in_channels=640))
runner = dict(type='EpochBasedRunner', max_epochs=100)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.25,
    step=[30, 60])
