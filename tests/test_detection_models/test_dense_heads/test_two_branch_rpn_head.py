# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import ConfigDict

from mmfewshot.detection.models.dense_heads import TwoBranchRPNHead


def test_attention_rpn_head():
    # Tests attention_rpn loss when truth is empty and non-empty.
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    config = ConfigDict(
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        mid_channels=64,
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        train_cfg=dict(
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
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7, offset=1),
            min_bbox_size=0))
    proposal_cfg = ConfigDict(
        nms_pre=2000,
        max_per_img=None,
        max_per_batch=2000,
        nms=dict(type='nms', iou_threshold=0.7, offset=1),
        min_bbox_size=0)

    self = TwoBranchRPNHead(**config)
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    feat = [
        torch.rand(1, 256, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    auxiliary_rpn_feats = [
        torch.rand(1, 256, s // 8, s // 8),
        torch.rand(1, 256, s // 16, s // 16)
    ]
    losses, proposal_list = self.forward_train(
        feat,
        auxiliary_rpn_feats,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        proposal_cfg=proposal_cfg)

    assert sum(losses['loss_rpn_cls']).item() > 0
    assert sum(losses['loss_rpn_cls_auxiliary']).item() > 0
    assert sum(losses['loss_rpn_bbox']).item() > 0
    assert len(proposal_list) == 1
    assert proposal_list[0].size(0) <= 2000
    losses, proposal_list = self.forward_train(
        feat,
        auxiliary_rpn_feats,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        proposal_cfg=proposal_cfg)

    assert sum(losses['loss_rpn_cls']).item() > 0
    assert sum(losses['loss_rpn_cls_auxiliary']).item() > 0
    assert sum(losses['loss_rpn_bbox']).item() > 0
    assert len(proposal_list) == 1
    assert proposal_list[0].size(0) <= 2000
