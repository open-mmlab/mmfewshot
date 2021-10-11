_base_ = ['meta_baseline_cub_5way_1shot_84x84_aug.py']

model = dict(
    type='MetaBaselineClassifier',
    backbone=dict(type='Conv4'),
    head=dict(type='MetaBaselineHead'))
load_from = './work_dirs/baseline_conv4_cub_5way_1shot/best_accuracy_mean.pth'
