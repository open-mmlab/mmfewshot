_base_ = ['relation_net_mini_imagenet_5way_1shot_84x84_aug.py']
model = dict(
    type='RelationNetClassifier',
    backbone=dict(type='Conv4NoPool'),
    head=dict(type='RelationHead', in_channels=64, feature_size=(19, 19)))
