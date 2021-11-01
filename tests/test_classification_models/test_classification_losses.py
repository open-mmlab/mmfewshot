# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.classification.models import build_loss


def test_mse_loss():
    cls_score = torch.Tensor([1, 1, 1, 1, 1, 0])
    label = torch.Tensor([1, 0, 1, 0, 1, 0])

    loss_cfg = dict(type='MSELoss', reduction='mean', loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(1 / 3))


def test_nll_loss():
    cls_score = torch.Tensor([[1, 0, 0], [1, 1, 0]])
    label = torch.LongTensor([1, 0])

    loss_cfg = dict(type='NLLLoss', reduction='mean', loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(-0.5000))
