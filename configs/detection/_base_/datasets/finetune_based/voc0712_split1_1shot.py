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

split = 1
all_classes = ALL_CLASSES[split]
base_classes = BASE_CLASSES[split]
novel_classes = NOVEL_CLASSES[split]
num_base_shot = 1
num_novel_shot = 1
# load few shot data :
# each ann file corresponding to one class
# all file should use same image prefix
ann_file_root = 'data/few_shot_voc_split/'
ann_file_per_class = []  # file path
img_prefix_per_class = []  # image prefix
ann_shot_filter_per_class = []  # ann filter for each ann file

for class_name in base_classes:
    ann_file_per_class.append(
        ann_file_root +
        f'{num_base_shot}shot/box_{num_base_shot}shot_{class_name}_train.txt')
    img_prefix_per_class.append(data_root)
    ann_shot_filter_per_class.append({class_name: num_base_shot})

for class_name in novel_classes:
    ann_file_per_class.append(
        ann_file_root +
        f'{num_novel_shot}shot/box_{num_novel_shot}shot_{class_name}_train.txt'
    )
    img_prefix_per_class.append(data_root)
    ann_shot_filter_per_class.append({class_name: num_novel_shot})

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 480), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='FewShotVOCDataset',
            ann_file=ann_file_per_class,
            img_prefix=img_prefix_per_class,
            ann_masks=ann_shot_filter_per_class,
            pipeline=train_pipeline,
            classes=all_classes,
            merge_dataset=True)),
    val=dict(
        type='VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=novel_classes),
    test=dict(
        type='VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=novel_classes))
evaluation = dict(interval=1, metric='mAP')
