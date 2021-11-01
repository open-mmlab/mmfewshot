# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.detection.models.roi_heads.bbox_heads import (
    ContrastiveBBoxHead, CosineSimBBoxHead, MetaBBoxHead,
    MultiRelationBBoxHead, TwoBranchBBoxHead)


def test_cosine_sim_bbox_head():
    self = CosineSimBBoxHead(num_shared_fcs=2, num_classes=20, scale=20)
    feat = torch.rand(20, 256, 7, 7)
    cls_scores, bbox_preds = self.forward(feat)
    assert cls_scores.shape == torch.Size([20, 21])
    assert bbox_preds.shape == torch.Size([20, 80])


def test_contrastive_bbox_head():
    self = ContrastiveBBoxHead(
        num_shared_fcs=2,
        num_classes=20,
        mlp_head_channels=128,
        with_weight_decay=True)
    feat = torch.rand(20, 256, 7, 7)
    cls_scores, bbox_preds, cont_feat = self.forward(feat)
    assert cls_scores.shape == torch.Size([20, 21])
    assert bbox_preds.shape == torch.Size([20, 80])
    assert cont_feat.shape == torch.Size([20, 128])
    feat = torch.Tensor([[1, 0, 1, 1], [1, 1, 1, 1]] * 2)
    label = torch.LongTensor([0, 1] * 2)
    iou = torch.Tensor([0.9] * 4)
    losses = self.loss_contrast(feat, label, iou)
    assert torch.allclose(losses['loss_contrast'], torch.tensor(5.0001))
    self.set_decay_rate(0.1)
    assert self._decay_rate == 0.1


def test_meta_bbox_head():
    self = MetaBBoxHead(num_meta_classes=20)
    feat = torch.rand(20, 2048)
    cls_scores = self.forward_meta_cls(feat)
    assert cls_scores.shape == torch.Size([20, 20])
    losses = self.loss_meta(cls_scores, torch.LongTensor([1] * 20),
                            torch.Tensor([1.0] * 20))
    assert 'meta_acc' in losses
    assert losses['loss_meta_cls'].item() > 0


def test_multi_relation_bbox_head():
    self = MultiRelationBBoxHead(in_channels=2048)
    query_feat = torch.rand(8, 2048, 7, 7)
    support_feat = torch.rand(1, 2048, 7, 7)
    cls_scores, bbox_pre = self(query_feat, support_feat)
    assert cls_scores.shape == torch.Size([8, 2])
    assert bbox_pre.shape == torch.Size([8, 4])


def test_two_branch_bbox_head():
    self = TwoBranchBBoxHead(
        reg_class_agnostic=True,
        num_classes=20,
        roi_feat_size=8,
        num_cls_fcs=2,
        num_reg_fcs=2,
        fc_out_channels=1024,
        auxiliary_loss_weight=0.1,
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    feat = torch.rand(8, 256, 8, 8)
    cls_scores, bbox_pre = self(feat)
    assert cls_scores.shape == torch.Size([8, 21])
    assert bbox_pre.shape == torch.Size([8, 4])
    cls_scores = self.forward_auxiliary(
        [torch.rand(2, 256, 8, 8),
         torch.rand(2, 256, 8, 8)])
    assert len(cls_scores[0]) == 2
    assert cls_scores[0][0].shape == torch.Size([2, 21])
