# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.detection.models.backbones import ResNetWithMetaConv


def test_resnet_with_meta_conv():
    model = ResNetWithMetaConv(depth=50, norm_eval=True)
    model.init_weights()
    model.train()
    # test w/o meat conv
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])
    # test w/ meat conv
    imgs = torch.randn(1, 4, 224, 224)
    feat = model(imgs, use_meta_conv=True)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])
