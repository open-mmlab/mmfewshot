# Copyright (c) 2019 Western Digital Corporation or its affiliates.
import copy

import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS

# AGGREGATORS are used for aggregate features from
# different data pipelines in meta-learning methods.
AGGREGATORS = MODELS


def build_aggregator(cfg):
    """Build aggregator."""
    return AGGREGATORS.build(cfg)


@AGGREGATORS.register_module()
class AggregationLayer(BaseModule):
    """Aggregate query and support features with single or multiple aggregator.
    Each aggregator return aggregated results in different way.

    Args:
        aggregator_cfgs (list[dict]): List of fusion function.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, aggregator_cfgs, init_cfg=None):
        super(AggregationLayer, self).__init__(init_cfg)
        self.aggregator_list = nn.ModuleList()
        self.num_aggregators = len(aggregator_cfgs)
        aggregator_cfgs_ = copy.deepcopy(aggregator_cfgs)
        for cfg in aggregator_cfgs_:
            self.aggregator_list.append(build_aggregator(cfg))

    def forward(self, query_feat, support_feat):
        """Return aggregated features of query and support through single or
        multiple aggregators.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with
                shape (N, C, H, W).

        Returns:
            list[Tensor]: List of aggregated features.
        """
        out = []
        for i in range(self.num_aggregators):
            out.append(self.aggregator_list[i](query_feat, support_feat))
        return out


@AGGREGATORS.register_module()
class DepthWiseCorrelationAggregator(BaseModule):
    """Depth-wise correlation aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 with_fc=False,
                 init_cfg=None):
        super(DepthWiseCorrelationAggregator, self).__init__(init_cfg)
        assert in_channels is not None, \
            "DepthWiseCorrelationAggregator require config of 'in_channels'."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_fc = with_fc
        if with_fc:
            assert out_channels is not None, 'out_channels is expected.'
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat, support_feat):
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (1, C, 1, 1).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        assert support_feat.size(0) == 1, \
            'first dimension of support features larger than 1'
        feat = F.conv2d(
            query_feat,
            support_feat.permute(1, 0, 2, 3),
            groups=self.in_channels)
        if self.with_fc:
            feat = self.fc(feat.squeeze())
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DifferenceAggregator(BaseModule):
    """Difference aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 with_fc=False,
                 init_cfg=None):
        super(DifferenceAggregator, self).__init__(init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat, support_feat):
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        assert support_feat.size(0) == 1, \
            'first dimension of support features larger than 1'
        feat = query_feat - support_feat
        if self.with_fc:
            feat = self.fc(feat.squeeze())
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DotProductAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 with_fc=False,
                 init_cfg=None):
        super(DotProductAggregator, self).__init__(init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat, support_feat):
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        feat = query_feat.mul(support_feat)
        if self.with_fc:
            feat = self.fc(feat.squeeze())
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat
