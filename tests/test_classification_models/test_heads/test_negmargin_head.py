# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmfewshot.classification.models.heads import NegMarginHead


@pytest.mark.parametrize('metric_type', ['cosine', 'softmax'])
def test_cosine_distance_head(metric_type):
    head = NegMarginHead(
        num_classes=100, in_channels=64, metric_type=metric_type)
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

    # test deep copy
    copy_head = copy.deepcopy(head)
    losses = copy_head.forward_train(feat, label)
    assert losses['loss'].item() > 0

    copy_head.before_forward_support()
    losses = copy_head.forward_support(feat, label)
    assert losses['loss'].item() > 0

    copy_head.before_forward_query()

    pred = copy_head.forward_query(feat)
    assert len(pred) == 4
    assert pred[0].shape[0] == 100
