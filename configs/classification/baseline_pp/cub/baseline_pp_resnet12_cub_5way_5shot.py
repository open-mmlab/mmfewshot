_base_ = ['baseline_pp_cub_5way_5shot_84x84_aug.py']

model = dict(
    type='BaselinePPClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(
        type='CosineDistanceHead',
        num_classes=100,
        in_channels=640,
        temperature=10.0),
    meta_test_head=dict(
        type='CosineDistanceHead',
        num_classes=5,
        in_channels=640,
        temperature=5.0))
