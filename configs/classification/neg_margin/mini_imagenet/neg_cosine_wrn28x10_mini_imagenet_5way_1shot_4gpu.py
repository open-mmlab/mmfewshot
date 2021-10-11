_base_ = ['neg_margin_mini_imagenet_5way_1shot_84x84_aug.py']

data = dict(samples_per_gpu=32, workers_per_gpu=2)
model = dict(
    type='NegMarginClassifier',
    backbone=dict(type='WRN28x10'),
    head=dict(
        type='NegMarginHead',
        num_classes=100,
        in_channels=640,
        metric_type='cosine',
        margin=-0.005,
        temperature=10.0),
    meta_test_head=dict(
        type='NegMarginHead',
        num_classes=5,
        in_channels=640,
        metric_type='cosine',
        margin=0.0,
        temperature=5.0))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
