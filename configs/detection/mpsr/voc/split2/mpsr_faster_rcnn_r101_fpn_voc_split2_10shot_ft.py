_base_ = [
    '../../../_base_/datasets/two_branch/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../mpsr_faster_rcnn_r101_fpn.py', '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    save_dataset=True,
    train=dict(
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MPSR', setting='SPLIT2_10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
            classes='ALL_CLASSES_SPLIT2')),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=500, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=2000)
optimizer = dict(
    lr=0.005,
    paramwise_cfg=dict(
        custom_keys={'.bias': dict(lr_mult=2.0, decay_mult=0.0)}))
lr_config = dict(warmup_iters=500, step=[1300, 1700])
runner = dict(max_iters=2000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'mpsr_faster_rcnn_r101_fpn_voc_split2_base_training/' \
    'latest.pth'
model = dict(
    roi_head=dict(
        bbox_head=dict(init_cfg=[
            dict(
                type='Normal',
                override=dict(type='Normal', name='fc_cls', std=0.001))
        ])))
