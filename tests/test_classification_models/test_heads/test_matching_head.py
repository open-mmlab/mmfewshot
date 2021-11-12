# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

from mmfewshot.classification.models.heads import MatchingHead


def test_matching_head():
    head = MatchingHead()
    support_feat = torch.randn(4, 64)
    query_feat = torch.randn(4, 64)
    label = torch.LongTensor([0, 1, 2, 3])

    losses = head.forward_train(support_feat, label, query_feat, label)
    assert losses['loss'].item() > 0

    head.before_forward_support()
    assert head.support_feats_list == []
    assert head.support_labels_list == []
    assert head.class_ids is None
    head.forward_support(support_feat, label)
    head.forward_support(support_feat, label)
    assert len(head.support_feats_list) == 2
    assert len(head.support_labels_list) == 2

    head.before_forward_query()
    assert head.class_ids is not None

    pred = head.forward_query(query_feat)
    assert len(pred) == 4
    assert pred[0].shape[0] == 4

    label = torch.LongTensor([0, 1, 2, 5])
    head.before_forward_support()
    assert head.support_feats_list == []
    assert head.support_labels_list == []
    assert head.class_ids is None
    head.forward_support(support_feat, label)
    head.forward_support(support_feat, label)
    assert len(head.support_feats_list) == 2
    assert len(head.support_labels_list) == 2
    with pytest.warns(None) as record:
        head.before_forward_query()
        assert len(record) > 0
