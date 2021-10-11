_base_ = ['matching_net_cub_5way_5shot_84x84_aug.py']
model = dict(
    type='MatchingNetClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(type='MatchingHead'))
