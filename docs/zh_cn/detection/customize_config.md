# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.
The detection part of mmfewshot is built upon the [mmdet](https://github.com/open-mmlab/mmdetection),
thus it is highly recommended learning the basic of [mmdet](https://mmdetection.readthedocs.io/en/latest/).


## Modify a config through script arguments

When submitting jobs using "tools/train.py" or "tools/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark \" is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config File Naming Convention

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{dataset}_{data setting}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `faster_rcnn`, `mask_rcnn`, etc.
- `[model setting]`: specific setting for some model, like `contrastive-loss` for `fsce`, etc.
- `{backbone}`: backbone type like `r50` (ResNet-50), `x101` (ResNeXt-101).
- `{neck}`: neck type like `fpn`,  `c4`.
- `[norm_setting]`: `bn` (Batch Normalization) is used unless specified, other norm layer type could be `gn` (Group Normalization), `syncbn` (Synchronized Batch Normalization).
    `gn-head`/`gn-neck` indicates GN is applied in head/neck only, while `gn-all` means GN is applied in the entire model, e.g. backbone, neck, head.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dconv`, `gcb`, `attention`, `albu`, `mstrain`.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8xb2` is used by default.
- `{dataset}`: dataset like `coco`, `voc-split1`, `voc-split2` and `voc-split3`.
- `{data setting}`: like `base-training` or `1shot-fine-tuning`.

## An Example of TFA

To help the users have a basic idea of a complete config and the modules in a modern classification system,
we make brief comments on the config of TFA in coco 10 shot fine-tuning setting as the following.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation.

```python
train_pipeline = [  # Training pipeline
    # First pipeline to load images from file path
    dict(type='LoadImageFromFile'),
    # Second pipeline to load annotations for current image
    dict(type='LoadAnnotations', with_bbox=True),
    # Augmentation pipeline that resize the images and their annotations
    dict(type='Resize',
         # The multiple scales of image
         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                    (1333, 768), (1333, 800)],
         # whether to keep the ratio between height and width
         keep_ratio=True,
         # the scales will be sampled from img_scale
         multiscale_mode='value'),
    # RandomFlip config, flip_ratio: the ratio or probability to flip
    dict(type='RandomFlip', flip_ratio=0.5),
    # Image normalization config to normalize the input images
    dict(type='Normalize',
         # Mean values used to in pre-trained backbone models
         mean=[103.53, 116.28, 123.675],
         # Standard variance used to in pre-trained backbone models
         std=[1.0, 1.0, 1.0],
         # The channel orders of image used in pre-trained backbone models
         to_rgb=False),
    # Padding config, size_divisor: the number the padded images should be divisible
    dict(type='Pad', size_divisor=32),
    # Default format bundle to gather data in the pipeline
    dict(type='DefaultFormatBundle'),
    # Pipeline that decides which keys in the data should be passed to the detector
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [  # test pipeline
    # First pipeline to load images from file path
    dict(type='LoadImageFromFile'),
    # An encapsulation that encapsulates the testing augmentations
    dict(type='MultiScaleFlipAug',
         # Decides the largest scale for testing, used for the Resize pipeline
         img_scale=(1333, 800),
         flip=False,  # Whether to flip images during testing
         transforms=[
             # Use resize augmentation
             dict(type='Resize', keep_ratio=True),
             # Augmentation pipeline that flip the images and their annotations
             dict(type='RandomFlip'),
             # Augmentation pipeline that normalize the input images
             dict(type='Normalize',
                  # Mean values used in pre-trained backbone models
                  mean=[103.53, 116.28, 123.675],
                  # Standard variance used in pre-trained backbone models
                  std=[1.0, 1.0, 1.0],
                  # The channel orders of image used in pre-trained backbone models
                  to_rgb=False),
             # Padding config, size_divisor: the number the padded images should be divisible
             dict(type='Pad', size_divisor=32),
             # Default format bundle to gather data in the pipeline
             dict(type='ImageToTensor', keys=['img']),
             # Pipeline that decides which keys in the data should be passed to the detector
             dict(type='Collect', keys=['img'])
         ])
]
data = dict(
    # Batch size of a single GPU
    samples_per_gpu=2,
    # Worker to pre-fetch data for each single GPU
    workers_per_gpu=2,
    train=dict(  # Train dataset config
        save_dataset=False,  # whether to save data_information into json file
        # the pre-defined few shot setting are saved in `FewShotCocoDefaultDataset`
        type='FewShotCocoDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='10SHOT')],  # pre-defined few shot setting
        img_prefix='data/coco/',  # prefix of images
        num_novel_shots=10,  # the max number of instances for novel classes
        num_base_shots=10,  # the max number of instances for base classes
        pipeline=train_pipeline,  # training pipeline
        classes='ALL_CLASSES',  # pre-defined classes split saved in dataset
        # whether to split the annotation (each image only contains one instance)
        instance_wise=False),
    val=dict(  # Validation dataset config
        type='FewShotCocoDataset',  # type of dataset
        ann_cfg=[dict(type='ann_file',  # type of ann_file
                      # path to ann_file
                      ann_file='data/few_shot_ann/coco/annotations/val.json')],
        # prefix of image
        img_prefix='data/coco/',
        pipeline=test_pipeline,  # testing pipeline
        classes='ALL_CLASSES'),
    test=dict(  # Testing dataset config
        type='FewShotCocoDataset',  # type of dataset
        ann_cfg=[dict(type='ann_file',  # type of ann_file
                      # path to ann_file
                      ann_file='data/few_shot_ann/coco/annotations/val.json')],
        # prefix of image
        img_prefix='data/coco/',
        pipeline=test_pipeline,  # testing pipeline
        test_mode=True,  # indicate in test mode
        classes='ALL_CLASSES'))  # pre-defined classes split saved in dataset
# The config to build the evaluation hook, refer to
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7
# for more details.
evaluation = dict(
    interval=80000,  # Evaluation interval
    metric='bbox',  # Metrics used during evaluation
    classwise=True,  # whether to show result of each class
    # eval results in pre-defined split of classes
    class_splits=['BASE_CLASSES', 'NOVEL_CLASSES'])
# Config used to build optimizer, support all the optimizers
# in PyTorch whose arguments are also the same as those in PyTorch
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# Config used to build the optimizer hook, refer to
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8
# for implementation details. Most of the methods do not use gradient clip.
optimizer_config = dict(grad_clip=None)
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(
    # The policy of scheduler, also support CosineAnnealing, Cyclic, etc.
    # Refer to details of supported LrUpdater from
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    policy='step',
    # The warmup policy, also support `exp` and `constant`.
    warmup='linear',
    # The number of iterations for warmup
    warmup_iters=10,
    # The ratio of the starting learning rate used for warmup
    warmup_ratio=0.001,
    # Steps to decay the learning rate
    step=[144000])
# Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
runner = dict(type='IterBasedRunner', max_iters=160000)
model = dict(  # The config of backbone
    type='TFA',  # The name of detector
    backbone=dict(
        type='ResNet',  # The name of detector
        # The depth of backbone, usually it is 50 or 101 for ResNet and ResNext backbones.
        depth=101,
        num_stages=4,  # Number of stages of the backbone.
        # The index of output feature maps produced in each stages
        out_indices=(0, 1, 2, 3),
        # The weights from stages 1 to 4 are frozen
        frozen_stages=4,
        # The config of normalization layers.
        norm_cfg=dict(type='BN', requires_grad=False),
        # Whether to freeze the statistics in BN
        norm_eval=True,
        # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv,
        # 'caffe' means stride 2 layers are in 1x1 convs.
        style='caffe'),
    neck=dict(
        # The neck of detector is FPN. For more details, please refer to
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py#L10
        type='FPN',
        # The input channels, this is consistent with the output channels of backbone
        in_channels=[256, 512, 1024, 2048],
        # The output channels of each level of the pyramid feature map
        out_channels=256,
        # The number of output scales
        num_outs=5,
        # the initialization of specific layer. For more details, please refer to
        # https://mmdetection.readthedocs.io/en/latest/tutorials/init_cfg.html
        init_cfg=[
            # initialize lateral_convs layer with Caffe2Xavier
            dict(type='Caffe2Xavier',
                 override=dict(type='Caffe2Xavier', name='lateral_convs')),
            # initialize fpn_convs layer with Caffe2Xavier
            dict(type='Caffe2Xavier',
                 override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]),
    rpn_head=dict(
        # The type of RPN head is 'RPNHead'. For more details, please refer to
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12
        type='RPNHead',
        # The input channels of each input feature map,
        # this is consistent with the output channels of neck
        in_channels=256,
        # Feature channels of convolutional layers in the head.
        feat_channels=256,
        anchor_generator=dict(  # The config of anchor generator
            # Most of methods use AnchorGenerator, For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10
            type='AnchorGenerator',
            # Basic scale of the anchor, the area of the anchor in one position
            # of a feature map will be scale * base_sizes
            scales=[8],
            # The ratio between height and width.
            ratios=[0.5, 1.0, 2.0],
            # The strides of the anchor generator. This is consistent with the FPN
            # feature strides. The strides will be taken as base_sizes if base_sizes is not set.
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(  # Config of box coder to encode and decode the boxes during training and testing
            # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods. For more details refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9
            type='DeltaXYWHBBoxCoder',
            # The target means used to encode and decode boxes
            target_means=[0.0, 0.0, 0.0, 0.0],
            # The standard variance used to encode and decode boxes
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # Config of loss function for the classification branch
        loss_cls=dict(
            # Type of loss for classification branch.
            type='CrossEntropyLoss',
            # RPN usually perform two-class classification,
            # so it usually uses sigmoid function.
            use_sigmoid=True,
            # Loss weight of the classification branch.
            loss_weight=1.0),
        # Config of loss function for the regression branch.
        loss_bbox=dict(
            # Type of loss, we also support many IoU Losses and smooth L1-loss. For implementation refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56
            type='L1Loss',
            # Loss weight of the regression branch.
            loss_weight=1.0)),
    roi_head=dict(
        # Type of the RoI head, for more details refer to
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10
        type='StandardRoIHead',
        # RoI feature extractor for bbox regression.
        bbox_roi_extractor=dict(
            # Type of the RoI feature extractor. For more details refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10
            type='SingleRoIExtractor',
            roi_layer=dict(  # Config of RoI Layer
                # Type of RoI Layer, for more details refer to
                # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79
                type='RoIAlign',
                output_size=7,  # The output size of feature maps.
                # Sampling ratio when extracting the RoI features.
                # 0 means adaptive ratio.
                sampling_ratio=0),
            # output channels of the extracted feature.
            out_channels=256,
            # Strides of multi-scale feature maps. It should be consistent to the architecture of the backbone.
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(  # Config of box head in the RoIHead.
            # Type of the bbox head, for more details refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177
            type='CosineSimBBoxHead',
            # Input channels for bbox head. This is consistent with the out_channels in roi_extractor
            in_channels=256,
            # Output feature channels of FC layers.
            fc_out_channels=1024,
            roi_feat_size=7,  # Size of RoI features
            num_classes=80,  # Number of classes for classification
            bbox_coder=dict(  # Box coder used in the second stage.
                # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods.
                type='DeltaXYWHBBoxCoder',
                # Means used to encode and decode box
                target_means=[0.0, 0.0, 0.0, 0.0],
                # Standard variance for encoding and decoding. It is smaller since
                # the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,  # Whether the regression is class agnostic.
            loss_cls=dict(  # Config of loss function for the classification branch
                # Type of loss for classification branch, we also support FocalLoss etc.
                type='CrossEntropyLoss',
                use_sigmoid=False,  # Whether to use sigmoid.
                loss_weight=1.0),  # Loss weight of the classification branch.
            loss_bbox=dict(  # Config of loss function for the regression branch.
                # Type of loss, we also support many IoU Losses and smooth L1-loss, etc.
                type='L1Loss',
                # Loss weight of the regression branch.
                loss_weight=1.0),
            # the initialization of specific layer. For more details, please refer to
            # https://mmdetection.readthedocs.io/en/latest/tutorials/init_cfg.html
            init_cfg=[
                # initialize shared_fcs layer with Caffe2Xavier
                dict(type='Caffe2Xavier',
                     override=dict(type='Caffe2Xavier', name='shared_fcs')),
                # initialize fc_cls layer with Normal
                dict(type='Normal',
                     override=dict(type='Normal', name='fc_cls', std=0.01)),
                # initialize fc_cls layer with Normal
                dict(type='Normal',
                     override=dict(type='Normal', name='fc_reg', std=0.001))
            ],
            # number of shared fc layers
            num_shared_fcs=2)),
    train_cfg=dict(
        rpn=dict(  # Training config of rpn
            assigner=dict(  # Config of assigner
                # Type of assigner. For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,  # IoU >= threshold 0.7 will be taken as positive samples
                neg_iou_thr=0.3,  # IoU < threshold 0.3 will be taken as negative samples
                min_pos_iou=0.3,  # The minimal IoU threshold to take boxes as positive samples
                # Whether to match the boxes under low quality (see API doc for more details).
                match_low_quality=True,
                ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
            sampler=dict(  # Config of positive/negative sampler
                # Type of sampler. For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8
                type='RandomSampler',
                num=256,  # Number of samples
                pos_fraction=0.5,  # The ratio of positive samples in the total samples.
                # The upper bound of negative samples based on the number of positive samples.
                neg_pos_ub=-1,
                # Whether add GT as proposals after sampling.
                add_gt_as_proposals=False),
            # The border allowed after padding for valid anchors.
            allowed_border=-1,
            # The weight of positive samples during training.
            pos_weight=-1,
            debug=False),  # Whether to set the debug mode
        rpn_proposal=dict(  # The config to generate proposals during training
            nms_pre=2000,  # The number of boxes before NMS
            max_per_img=1000,  # The number of boxes to be kept after NMS.
            nms=dict(  # Config of NMS
                type='nms',  # Type of NMS
                iou_threshold=0.7),  # NMS threshold
            min_bbox_size=0),  # The allowed minimal box size
        rcnn=dict(  # The config for the roi heads.
            assigner=dict(  # Config of assigner for second stage, this is different for that in rpn
                # Type of assigner, MaxIoUAssigner is used for all roi_heads for now. For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,  # IoU >= threshold 0.5 will be taken as positive samples
                neg_iou_thr=0.5,  # IoU < threshold 0.5 will be taken as negative samples
                min_pos_iou=0.5,  # The minimal IoU threshold to take boxes as positive samples
                # Whether to match the boxes under low quality (see API doc for more details).
                match_low_quality=False,
                ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
            sampler=dict(
                # Type of sampler, PseudoSampler and other samplers are also supported. For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8
                type='RandomSampler',
                num=512,  # Number of samples
                pos_fraction=0.25,  # The ratio of positive samples in the total samples.
                # The upper bound of negative samples based on the number of positive samples.
                neg_pos_ub=-1,
                # Whether add GT as proposals after sampling.
                add_gt_as_proposals=True),
            # The weight of positive samples during training.
            pos_weight=-1,
            # Whether to set the debug mode
            debug=False)),
    test_cfg=dict(  # Config for testing hyperparameters for rpn and rcnn
        rpn=dict(  # The config to generate proposals during testing
            # The number of boxes before NMS
            nms_pre=1000,
            # The number of boxes to be kept after NMS.
            max_per_img=1000,
            # Config of NMS
            nms=dict(type='nms', iou_threshold=0.7),
            # The allowed minimal box size
            min_bbox_size=0),
        rcnn=dict(  # The config for the roi heads.
            score_thr=0.05,  # Threshold to filter out boxes
            # Config of NMS in the second stage
            nms=dict(type='nms', iou_threshold=0.5),
            # Max number of detections of each image
            max_per_img=100)),
    # parameters with the prefix listed in frozen_parameters will be frozen
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
    ])
# Config to set the checkpoint hook, Refer to
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
checkpoint_config = dict(interval=80000)
# The logger used to record the training process.
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]  # cumstomize hook
dist_params = dict(backend='nccl')  # parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # the output level of the log.
# use base training model as model initialization.
load_from = 'work_dirs/tfa_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth'
# workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
workflow = [('train', 1)]
use_infinite_sampler = True  # whether to use infinite sampler
seed = 0  # random seed
```

## FAQ

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, user need to pass the intermediate variables into corresponding fields again.
For example, we would like to use multi scale strategy to train a Mask R-CNN. `train_pipeline`/`test_pipeline` are intermediate variable we would like modify.

```python
_base_ = './faster_rcnn_r50_caffe_fpn.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode="value",
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
```

We first define the new `train_pipeline`/`test_pipeline` and pass them into `data`.

Similarly, if we would like to switch from `SyncBN` to `BN` or `MMSyncBN`, we need to substitute every `norm_cfg` in the config.

```python
_base_ = './faster_rcnn_r50_caffe_fpn.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    ...)
```
