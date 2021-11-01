_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
]
num_support_ways = 2
num_support_shots = 10
# model settings
model = dict(
    type='AttentionRPNDetector',
    backbone=dict(frozen_stages=2),
    rpn_head=dict(
        type='AttentionRPNHead',
        in_channels=1024,
        feat_channels=1024,
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=1024,
                    with_fc=False)
            ])),
    roi_head=dict(
        type='MultiRelationRoIHead',
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MultiRelationBBoxHead',
            patch_relation=True,
            local_correlation=True,
            global_relation=True,
            roi_feat_size=14,
            in_channels=2048,
            num_classes=1,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            init_cfg=[
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='patch_relation_branch',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='patch_relation_fc_cls',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='patch_relation_fc_reg',
                        std=0.001)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal',
                        name='local_correlation_branch',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal',
                        name='local_correlation_fc_cls',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='global_relation_branch',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='global_relation_fc_cls',
                        std=0.01)),
            ])),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
