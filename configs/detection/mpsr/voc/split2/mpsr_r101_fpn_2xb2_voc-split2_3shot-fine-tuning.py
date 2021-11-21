_base_ = [
    '../../../_base_/datasets/two_branch/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../mpsr_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MPSR', setting='SPLIT2_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT2')),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=500, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=2000)
optimizer = dict(
    lr=0.005,
    paramwise_cfg=dict(
        custom_keys=dict({'.bias': dict(lr_mult=2.0, decay_mult=0.0)})))
lr_config = dict(
    warmup_iters=500,
    warmup_ratio=1. / 3,
    step=[1300],
)
runner = dict(max_iters=2000)
# load_from = 'path of base training model'
load_from = (
    'work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/latest.pth')
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(roi_layer=dict(aligned=False)),
        bbox_head=dict(init_cfg=[
            dict(
                type='Normal',
                override=dict(type='Normal', name='fc_cls', std=0.001))
        ])))
