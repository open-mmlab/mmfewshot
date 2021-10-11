_base_ = ['proto_net_cub_5way_5shot_84x84_aug.py']

model = dict(
    type='ProtoNetClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(type='PrototypeHead'))
