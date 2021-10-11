_base_ = ['proto_net_cub_5way_1shot_84x84_aug.py']

model = dict(
    type='ProtoNetClassifier',
    backbone=dict(type='Conv4'),
    head=dict(type='PrototypeHead'))
