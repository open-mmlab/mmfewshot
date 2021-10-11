_base_ = ['meta_baseline_tiered_imagenet_5way_1shot_84x84_aug.py']

model = dict(
    type='MetaBaselineClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(type='MetaBaselineHead'))
load_from = './work_dirs/baseline_resnet12_tiered_imagenet_5way_1shot/' \
            'best_accuracy_mean.pth'
