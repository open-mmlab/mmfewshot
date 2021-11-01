# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.classification.models.heads import LinearHead


def test_linear_head():
    head = LinearHead(num_classes=100, in_channels=64)
    feat = torch.randn(4, 64)
    label = torch.LongTensor([0, 1, 2, 3])

    losses = head.forward_train(feat, label)
    assert losses['loss'].item() > 0

    head.before_forward_support()
    losses = head.forward_support(feat, label)
    assert losses['loss'].item() > 0

    head.before_forward_query()

    pred = head.forward_query(feat)
    assert len(pred) == 4
    assert pred[0].shape[0] == 100
