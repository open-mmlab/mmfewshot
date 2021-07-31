_base_ = [
    '../../../_base_/datasets/finetune_based/base_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(dataset=dict(classes='BASE_CLASSES_SPLIT3')),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'))
lr_config = dict(warmup_iters=100, step=[12000, 16000])
runner = dict(max_iters=18000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=15)))
