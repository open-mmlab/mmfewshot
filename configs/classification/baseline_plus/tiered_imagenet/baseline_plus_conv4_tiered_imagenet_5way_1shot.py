_base_ = ['baseline_pp_tiered_imagenet_5way_1shot_84x84_aug.py']

model = dict(
    type='BaselinePlusClassifier',
    backbone=dict(type='Conv4'),
    head=dict(type='CosineDistanceHead', num_classes=351, in_channels=1600),
    meta_test_head=dict(
        type='CosineDistanceHead', num_classes=5, in_channels=1600))
