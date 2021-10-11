_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../fsdetview_faster_rcnn_r50_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=30,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='FSDetView', setting='30SHOT')],
            num_novel_shots=30,
            num_base_shots=30,
        )),
    model_init=dict(num_novel_shots=30, num_base_shots=30))
evaluation = dict(interval=1000)
checkpoint_config = dict(interval=1000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[8000])
runner = dict(max_iters=8000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'fsdetview_faster_rcnn_r50_c4_coco_base_training/' \
    'latest.pth'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
])
