_base_ = ['baseline_mini_imagenet_5way_5shot_84x84_aug.py']
model = dict(
    type='BaselineClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(type='LinearHead', num_classes=64, in_channels=640),
    meta_test_head=dict(type='LinearHead', num_classes=5, in_channels=640))
