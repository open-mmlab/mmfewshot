_base_ = [
    'maml_tiered_imagenet_5way_1shot_84x84_aug.py',
]
model = dict(
    type='MAMLClassifier',
    num_inner_steps=2,
    inner_lr=0.01,
    first_order=False,
    backbone=dict(type='ResNet12'),
    head=dict(type='LinearHead', num_classes=5, in_channels=640))
