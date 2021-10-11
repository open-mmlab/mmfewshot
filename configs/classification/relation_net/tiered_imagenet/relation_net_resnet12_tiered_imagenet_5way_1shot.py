_base_ = ['relation_net_tiered_imagenet_5way_1shot_84x84_aug.py']
model = dict(
    type='RelationNetClassifier',
    backbone=dict(type='ResNet12', with_avgpool=False, flatten=False),
    head=dict(type='RelationHead', in_channels=640, feature_size=(5, 5)))
