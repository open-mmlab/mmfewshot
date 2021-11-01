# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.classification.models.heads import MetaBaselineHead


def test_meta_baseline_head():
    head = MetaBaselineHead()
    support_feat = torch.randn(4, 64)
    query_feat = torch.randn(4, 64)
    label = torch.LongTensor([0, 1, 2, 3])

    losses = head.forward_train(support_feat, label, query_feat, label)
    assert losses['loss'].item() > 0

    head.before_forward_support()
    assert head.support_feats == []
    assert head.support_labels == []
    assert head.class_ids is None
    assert head.mean_support_feats is None
    head.forward_support(support_feat, label)
    head.forward_support(support_feat, label)
    assert len(head.support_feats) == 2
    assert len(head.support_labels) == 2

    head.before_forward_query()
    assert head.class_ids is not None
    assert head.mean_support_feats is not None

    pred = head.forward_query(query_feat)
    assert len(pred) == 4
    assert pred[0].shape[0] == 4
