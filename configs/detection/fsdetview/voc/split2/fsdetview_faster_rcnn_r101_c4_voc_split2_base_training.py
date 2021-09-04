_base_ = [
    '../../../_base_/datasets/nway_kshot/base_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../fsdetview_faster_rcnn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(
        save_dataset=False, dataset=dict(classes='BASE_CLASSES_SPLIT2')),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'),
    model_init=dict(classes='BASE_CLASSES_SPLIT2'))
lr_config = dict(warmup=None, step=[18000])
evaluation = dict(interval=24000)
checkpoint_config = dict(interval=6000)
runner = dict(max_iters=24000)
optimizer = dict(lr=0.001)
# model settings
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(num_classes=15, num_meta_classes=15),
        shared_head=dict(pretrained=pretrained),
    ))
