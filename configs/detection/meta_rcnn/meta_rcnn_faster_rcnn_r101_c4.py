_base_ = [
    './meta_rcnn_faster_rcnn_r50_c4.py',
]
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(num_classes=20, num_meta_classes=20)))
