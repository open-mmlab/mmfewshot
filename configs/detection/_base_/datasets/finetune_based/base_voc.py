# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                   (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data_root = 'data/VOCdevkit/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        save_dataset=False,
        dataset=dict(
            type='FewShotVOCDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root +
                    'VOC2007/ImageSets/Main/trainval.txt'),
                dict(
                    type='ann_file',
                    ann_file=data_root + 'VOC2012/ImageSets/Main/trainval.txt')
            ],
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=None,
            use_difficult=True,
            instance_wise=False)),
    val=dict(
        type='FewShotVOCDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=None,
    ),
    test=dict(
        type='FewShotVOCDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None,
    ))
evaluation = dict(interval=2000, metric='mAP')
