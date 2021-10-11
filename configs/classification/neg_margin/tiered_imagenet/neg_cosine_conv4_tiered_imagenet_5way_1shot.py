_base_ = ['neg_margin_tiered_imagenet_5way_1shot_84x84_aug.py']

model = dict(
    type='NegMarginClassifier',
    backbone=dict(type='Conv4'),
    head=dict(
        type='NegMarginHead',
        num_classes=351,
        in_channels=1600,
        metric_type='cosine',
        margin=-0.02,
        temperature=10.0),
    meta_test_head=dict(
        type='NegMarginHead',
        num_classes=5,
        in_channels=1600,
        metric_type='cosine',
        margin=0.0,
        temperature=5.0))
