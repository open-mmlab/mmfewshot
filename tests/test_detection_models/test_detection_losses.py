# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.detection.models import build_loss


def test_contrastive_loss():
    feat = torch.Tensor([[1, 0, 1, 1], [1, 1, 1, 1]] * 2)
    label = torch.LongTensor([0, 1] * 2)
    iou = torch.Tensor([0.9] * 4)

    loss_cfg = dict(
        type='SupervisedContrastiveLoss',
        temperature=0.2,
        iou_threshold=0.8,
        loss_weight=0.5,
        reweight_type='none')
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(feat, label, iou), torch.tensor(0.1390))
    loss = build_loss(loss_cfg)
    assert torch.allclose(
        loss(feat, label, iou, decay_rate=0.5), torch.tensor(0.1390 * 0.5))
    iou = torch.Tensor([0.7] * 4)
    assert torch.allclose(loss(feat, label, iou), torch.tensor(0.))

    loss_cfg = dict(
        type='SupervisedContrastiveLoss',
        temperature=0.2,
        iou_threshold=0.5,
        loss_weight=0.5,
        reweight_type='linear')
    loss = build_loss(loss_cfg)
    assert loss(feat, label, iou)

    loss_cfg = dict(
        type='SupervisedContrastiveLoss',
        temperature=0.2,
        iou_threshold=0.5,
        loss_weight=0.5,
        reweight_type='exp')
    loss = build_loss(loss_cfg)
    assert loss(feat, label, iou)
