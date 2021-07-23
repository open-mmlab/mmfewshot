# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1000, 440), (1000, 472), (1000, 504), (1000, 536),
                       (1000, 568), (1000, 600)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='AttentionRPNCropResizeSupport',
            context_pixel=16,
            target_size=(320, 320)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = 'data/coco/'
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    copy_random_support=False,
    train=dict(
        type='QueryAwareDataset',
        num_support_ways=2,
        num_support_shots=10,
        repeat_times=1,
        save_dataset=False,
        dataset=dict(
            type='FewShotCocoDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file='data/coco/annotations/instances_train2017.json')
            ],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='BASE_CLASSES',
            instance_wise=False,
            min_bbox_area_filter=32 * 32,
            dataset_name='query-support dataset')),
    val=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_val2017.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='BASE_CLASSES'),
    test=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_val2017.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES'),
    # random sample 10 shot base instance to evaluate training
    support_template=dict(
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_train2017.json')
        ],
        img_prefix='data/coco/train2017',
        pipeline=train_multi_pipelines['support'],
        classes='BASE_CLASSES',
        num_base_shots=10,
        instance_wise=True,
        min_bbox_area_filter=32 * 32,
        dataset_name='support template'))
evaluation = dict(interval=20000, metric='bbox', classwise=True)
