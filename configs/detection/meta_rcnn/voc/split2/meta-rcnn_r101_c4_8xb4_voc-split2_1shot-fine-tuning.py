_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../meta-rcnn_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT2_1SHOT')],
            num_novel_shots=1,
            num_base_shots=1,
            classes='ALL_CLASSES_SPLIT2',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'),
    model_init=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=50, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=50)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=100)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
])
