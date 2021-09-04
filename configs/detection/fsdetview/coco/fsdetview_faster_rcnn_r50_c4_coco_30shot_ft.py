_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../fsdetview_faster_rcnn_r50_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(
        save_dataset=True,
        repeat_times=10,
        num_used_support_shots=30,
        dataset=dict(
            num_novel_shots=30,
            num_base_shots=30,
        )),
    model_init=dict(num_novel_shots=30, num_base_shots=30))
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=3000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[4000])
runner = dict(max_iters=6000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'fsdetview_faster_rcnn_r50_c4_coco_base_training/' \
    'latest.pth'
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80, num_meta_classes=80)))
