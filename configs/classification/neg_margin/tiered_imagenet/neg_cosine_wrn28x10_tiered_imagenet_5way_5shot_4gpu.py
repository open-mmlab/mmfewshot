_base_ = ['neg_margin_tiered_imagenet_5way_5shot_84x84_aug.py']

data = dict(samples_per_gpu=32, workers_per_gpu=2)
model = dict(
    type='NegMarginClassifier',
    backbone=dict(type='WRN28x10'),
    head=dict(
        type='NegMarginHead',
        num_classes=351,
        in_channels=640,
        metric_type='cosine',
        margin=-0.02,
        temperature=10.0),
    meta_test_head=dict(
        type='NegMarginHead',
        num_classes=5,
        in_channels=640,
        metric_type='cosine',
        margin=0.0,
        temperature=5.0))
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.25,
    step=[60, 120])
