_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../fsdetview_faster_rcnn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            num_novel_shots=5,
            num_base_shots=5,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=1500,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=1500)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup=None, step=[
        1500,
    ])
runner = dict(max_iters=1500)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/' \
    'fsdetview_faster_rcnn_r101_c4_voc_split1_base_training/' \
    'latest.pth'
# model settings
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(num_classes=20, num_meta_classes=20),
        shared_head=dict(pretrained=pretrained)))
