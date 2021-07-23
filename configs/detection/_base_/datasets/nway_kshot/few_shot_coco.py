# dataset settings
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=(1000, 600),
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ResizeWithMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    copy_random_support=True,
    train=dict(
        type='NwayKshotDataset',
        num_support_ways=80,
        num_support_shots=1,
        mutual_support_shot=True,
        num_used_support_shots=None,
        save_dataset=True,
        repeat_times=10,
        dataset=dict(
            type='FewShotCocoDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file='data/few_shot_coco_split/'
                    'annotations/trainvalno5k.json')
            ],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='ALL_CLASSES',
            instance_wise=False,
            dataset_name='query-support dataset')),
    val=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_coco_split/annotations/5k.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='ALL_CLASSES'),
    test=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_coco_split/annotations/5k.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES'),
    support_template=dict(
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotCocoDataset',
        ann_cfg=None,
        img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        instance_wise=True,
        classes='ALL_CLASSES',
        dataset_name='support template'))
evaluation = dict(
    interval=3000,
    metric='bbox',
    classwise=True,
    class_splits=['BASE_CLASSES', 'NOVEL_CLASSES'])
