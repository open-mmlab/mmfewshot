ALL_CLASSES = {
    1: ('aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa'),
    2: ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    3: ('aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa')
}

NOVEL_CLASSES = {
    1: ('bird', 'bus', 'cow', 'motorbike', 'sofa'),
    2: ('aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    3: ('boat', 'cat', 'motorbike', 'sheep', 'sofa'),
}

BASE_CLASSES = {
    1: ('aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'),
    2: ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'),
    3: ('aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor')
}

# dataset settings
data_root = 'data/VOCdevkit/'

# few shot setting
split = 1
base_classes = BASE_CLASSES[split]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NwayKshotDataset',
        support_way=15,
        support_shot=1,
        dataset=dict(
            type='FewShotVOCDataset',
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root, data_root],
            pipeline=train_pipeline,
            classes=base_classes,
            merge_dataset=True,
        )),
    val=dict(
        type='VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=base_classes),
    test=dict(
        type='VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=base_classes))
evaluation = dict(interval=1, metric='mAP')
