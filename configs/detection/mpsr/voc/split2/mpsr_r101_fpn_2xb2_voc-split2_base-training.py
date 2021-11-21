"""!!! Currently, The base training models of MPSR are very unstable for few
shot fine-tuning. To reproduce the papers result, please use the converted
weights of base models from original implementation. The download link can be
found in /configs/detection/mpsr/README.md.

!!! We will continue to investigate and improve it.
"""

_base_ = [
    '../../../_base_/datasets/two_branch/base_voc.py',
    '../../../_base_/schedules/schedule.py', '../../mpsr_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(dataset=dict(classes='BASE_CLASSES_SPLIT2')),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'))
evaluation = dict(interval=9000, class_splits=['BASE_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=3000)
lr_config = dict(warmup_iters=500, warmup_ratio=1. / 3, step=[24000, 32000])
optimizer = dict(
    lr=0.005,
    paramwise_cfg=dict(
        custom_keys=dict({'.bias': dict(lr_mult=2.0, decay_mult=0.0)})))
runner = dict(max_iters=36000)
model = dict(roi_head=dict(bbox_head=dict(num_classes=15)))
