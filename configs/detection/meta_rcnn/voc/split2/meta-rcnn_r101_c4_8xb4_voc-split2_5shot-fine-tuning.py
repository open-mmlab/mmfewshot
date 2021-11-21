_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../meta-rcnn_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT2_5SHOT')],
            num_novel_shots=5,
            num_base_shots=5,
            classes='ALL_CLASSES_SPLIT2',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'),
    model_init=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=200, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=200)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=800)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
])
