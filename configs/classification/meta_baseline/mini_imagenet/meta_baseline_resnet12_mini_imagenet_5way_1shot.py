_base_ = ['meta_baseline_mini_imagenet_5way_1shot_84x84_aug.py']
model = dict(
    type='MetaBaselineClassifier',
    backbone=dict(type='ResNet12'),
    head=dict(type='MetaBaselineHead'))
load_from = './work_dirs/baseline_resnet12_mini_imagenet_5way_1shot/' \
            'best_accuracy_mean.pth'
