_base_ = ['neg_margin_mini_imagenet_5way_5shot_84x84_aug.py']

model = dict(
    type='NegMarginClassifier',
    backbone=dict(type='Conv4'),
    head=dict(
        type='NegMarginHead',
        num_classes=100,
        in_channels=1600,
        metric_type='cosine',
        margin=-0.005,
        temperature=10.0),
    meta_test_head=dict(
        type='NegMarginHead',
        num_classes=5,
        in_channels=1600,
        metric_type='cosine',
        margin=0.0,
        temperature=5.0))
