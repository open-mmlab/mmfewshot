_base_ = [
    '../../_base_/datasets/two_branch/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../mpsr_faster_rcnn_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MPSR', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10)))
evaluation = dict(interval=4000)
checkpoint_config = dict(interval=4000)
optimizer = dict(lr=0.005)
lr_config = dict(warmup_iters=500, warmup_ratio=1. / 3, step=[2800, 3500])
runner = dict(max_iters=4000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'mpsr_faster_rcnn_r101_fpn_coco_base_training/' \
    'latest.pth'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=80,
            init_cfg=[
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.001))
            ])))
