# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.detection.models.roi_heads import MetaRCNNResLayer


def test_resnet_with_meta_conv():
    layer = MetaRCNNResLayer(depth=50)
    layer.init_weights()
    layer.train()
    imgs = torch.randn(1, 1024, 32, 32)
    feat = layer(imgs)
    assert feat.shape == torch.Size([1, 2048])
