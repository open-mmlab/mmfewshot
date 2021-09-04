_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../meta_rcnn_faster_rcnn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT2',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'),
    model_init=dict(classes='BASE_CLASSES_SPLIT2'))
evaluation = dict(
    interval=900, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=900)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup=None, step=[
        600,
    ])
runner = dict(max_iters=900)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'meta_rcnn_faster_rcnn_r101_c4_voc_split2_base_training/' \
    'latest.pth'
# model settings
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(num_classes=20, num_meta_classes=20),
        shared_head=dict(pretrained=pretrained),
    ))
