# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import ConfigDict

from mmfewshot.detection.models.roi_heads import (ContrastiveRoIHead,
                                                  FSDetViewRoIHead,
                                                  MetaRCNNRoIHead,
                                                  MultiRelationRoIHead,
                                                  TwoBranchRoIHead)


def test_contrastive_roi_head():
    cfg = ConfigDict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='ContrastiveBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            num_shared_fcs=2,
            mlp_head_channels=128,
            with_weight_decay=True,
            loss_contrast=dict(
                type='SupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.6,
                loss_weight=0.2,
                reweight_type='none'),
            scale=20),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    self = ContrastiveRoIHead(**cfg)
    feat = [torch.rand(1, 256, s // i, s // i) for i in [4, 8, 16, 32, 64]]
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]
    bbox_results = self.forward_train(
        feat, img_metas, [torch.Tensor([[1., 1., 30., 30., 0.8]] * 500)],
        gt_bboxes, [torch.LongTensor([0])])
    assert 'loss_cls' in bbox_results
    assert 'acc' in bbox_results
    assert 'loss_bbox' in bbox_results
    assert 'loss_contrast' in bbox_results


def test_meta_rcnn_roi_head():
    cfg = ConfigDict(
        shared_head=dict(
            type='MetaRCNNResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MetaBBoxHead',
            with_avg_pool=False,
            roi_feat_size=1,
            in_channels=2048,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
            num_meta_classes=20,
            meta_cls_in_channels=2048,
            with_meta_cls_loss=True,
            loss_meta=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DotProductAggregator',
                    in_channels=2048,
                    with_fc=False)
            ]),
        train_cfg=dict(
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
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100))
    self = MetaRCNNRoIHead(**cfg)
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    query_feat = [torch.randn(1, 1024, 32, 32)]
    support_feat = [torch.randn(15, 1024, 14, 14)]
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]

    losses = self.forward_train(
        query_feat, support_feat,
        [torch.Tensor([[1., 1., 30., 30., 0.8]] * 500)], img_metas, gt_bboxes,
        [torch.LongTensor([0])], [torch.LongTensor([i]) for i in range(15)])
    assert 'loss_cls' in losses
    assert 'acc' in losses
    assert 'loss_bbox' in losses
    assert 'loss_meta_cls' in losses
    assert 'meta_acc' in losses

    results = self.simple_test(query_feat,
                               {i: torch.randn(1, 2048)
                                for i in range(20)},
                               [torch.Tensor([[1., 1., 30., 30., 0.8]] * 300)],
                               img_metas, False)
    assert len(results) == 1
    assert len(results[0]) == 20


def test_fsdetview_roi_head():
    cfg = ConfigDict(
        shared_head=dict(
            type='MetaRCNNResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MetaBBoxHead',
            with_avg_pool=False,
            roi_feat_size=1,
            in_channels=4096,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
            num_meta_classes=20,
            meta_cls_in_channels=2048,
            with_meta_cls_loss=True,
            loss_meta=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True),
                dict(
                    type='DifferenceAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True)
            ]),
        train_cfg=dict(
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
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100))
    self = FSDetViewRoIHead(**cfg)
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    query_feat = [torch.randn(1, 1024, 32, 32)]
    support_feat = [torch.randn(15, 1024, 14, 14)]
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]

    losses = self.forward_train(
        query_feat, support_feat,
        [torch.Tensor([[1., 1., 30., 30., 0.8]] * 500)], img_metas, gt_bboxes,
        [torch.LongTensor([0])], [torch.LongTensor([i]) for i in range(15)])
    assert 'loss_cls' in losses
    assert 'acc' in losses
    assert 'loss_bbox' in losses
    assert 'loss_meta_cls' in losses
    assert 'meta_acc' in losses


def test_multi_relation_roi_head():
    cfg = ConfigDict(
        shared_head=dict(
            type='ResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MultiRelationBBoxHead',
            with_avg_pool=True,
            roi_feat_size=14,
            in_channels=2048,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            patch_relation=True,
            local_correlation=True,
            global_relation=True),
        num_support_ways=2,
        num_support_shots=5,
        train_cfg=dict(
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
            debug=False),
        test_cfg=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    self = MultiRelationRoIHead(**cfg)
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    query_feat = [torch.rand(1, 1024, 32, 32)]
    support_feat = [torch.rand(10, 1024, 20, 20)]
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]

    losses = self.forward_train(
        query_feat, support_feat,
        [torch.Tensor([[1., 1., 30., 30., 0.8]] * 500)] * 2, img_metas,
        gt_bboxes, [torch.LongTensor([1])], gt_bboxes * 10)
    assert 'loss_cls' in losses
    assert 'acc' in losses
    assert 'loss_bbox' in losses
    results = self.simple_test(query_feat, torch.rand(1, 2048, 7, 7),
                               [torch.Tensor([[1., 1., 30., 30., 0.8]] * 100)],
                               img_metas)
    assert len(results) == 1
    assert len(results[0]) == 1


def test_two_branch_roi_head():
    cfg = ConfigDict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=8, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='TwoBranchBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=8,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0, beta=1.0),
            num_cls_fcs=2,
            num_reg_fcs=2,
            auxiliary_loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5, offset=1),
            max_per_img=100))
    self = TwoBranchRoIHead(**cfg)

    feat = [torch.rand(1, 256, 8, 8), torch.rand(1, 256, 8, 8)]
    auxiliary_losses = self.forward_auxiliary_train(
        feat, [torch.LongTensor([0])] * 2)
    assert 'loss_cls_auxiliary' in auxiliary_losses
    assert 'acc_auxiliary' in auxiliary_losses
