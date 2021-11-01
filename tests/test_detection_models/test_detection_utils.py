# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import ConfigDict

from mmfewshot.detection.models.utils import (AggregationLayer,
                                              DepthWiseCorrelationAggregator,
                                              DifferenceAggregator,
                                              DotProductAggregator)


def test_depth_wise_aggregator():
    # test forward w/o fc
    self = DepthWiseCorrelationAggregator(in_channels=256)
    query_feat = torch.randn(1, 256, 7, 7)
    support_feat = torch.randn(1, 256, 3, 3)
    out = self(query_feat, support_feat)
    assert out.shape == torch.Size([1, 256, 5, 5])
    # test forward w/ fc
    self = DepthWiseCorrelationAggregator(
        in_channels=256, out_channels=64, with_fc=True)
    query_feat = torch.randn(2, 256, 7, 7)
    support_feat = torch.randn(1, 256, 7, 7)
    out = self(query_feat, support_feat)
    assert out.shape == torch.Size([2, 64])


def test_diff_aggregator():
    # test forward w/o fc
    self = DifferenceAggregator(in_channels=256)
    query_feat = torch.randn(2, 256, 7, 7)
    support_feat = torch.randn(1, 256, 7, 7)
    out = self(query_feat, support_feat)
    assert out.shape == torch.Size([2, 256, 7, 7])
    # test forward w/ fc
    self = DifferenceAggregator(in_channels=256, out_channels=64, with_fc=True)
    query_feat = torch.randn(2, 256, 1, 1)
    support_feat = torch.randn(1, 256, 1, 1)
    out = self(query_feat, support_feat)
    assert out.shape == torch.Size([2, 64])


def test_dot_product_aggregator():
    # test forward w/o fc
    self = DotProductAggregator(in_channels=256)
    query_feat = torch.randn(2, 256, 7, 7)
    support_feat = torch.randn(1, 256, 7, 7)
    out = self(query_feat, support_feat)
    assert out.shape == torch.Size([2, 256, 7, 7])
    # test forward w/ fc
    self = DotProductAggregator(in_channels=256, out_channels=64, with_fc=True)
    query_feat = torch.randn(2, 256, 1, 1)
    support_feat = torch.randn(1, 256, 1, 1)
    out = self(query_feat, support_feat)
    assert out.shape == torch.Size([2, 64])


def test_aggregation_layer():
    cfg = ConfigDict(aggregator_cfgs=[
        dict(type='DepthWiseCorrelationAggregator', in_channels=256),
        dict(type='DifferenceAggregator', in_channels=256),
        dict(type='DotProductAggregator', in_channels=256),
    ])
    self = AggregationLayer(**cfg)
    query_feat = torch.randn(2, 256, 1, 1)
    support_feat = torch.randn(1, 256, 1, 1)
    out = self(query_feat, support_feat)
    assert len(out) == 3
    assert out[0].shape == torch.Size([2, 256, 1, 1])
    assert out[1].shape == torch.Size([2, 256, 1, 1])
    assert out[2].shape == torch.Size([2, 256, 1, 1])
