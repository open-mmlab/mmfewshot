# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmfewshot.classification.models.backbones.resnet12 import (BasicBlock,
                                                                ResNet12)


def test_basicblock():
    downsample = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64),
    )

    # BasicBlock w/o downsample
    block = BasicBlock(64, 64, 2, None, 0, False, 1)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)

    # BasicBlock w/ downsample

    block = BasicBlock(3, 64, 2, downsample, 0, False, 1)
    x = torch.randn(1, 3, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)

    # 3 BasicBlock w/ downsample and dropout
    block = BasicBlock(3, 64, 2, downsample, 0.1, False, 1)
    x = torch.randn(1, 3, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)

    # 3 BasicBlock w/ downsample and drop block
    block = BasicBlock(3, 64, 2, downsample, 0, True, 5)
    x = torch.randn(1, 3, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)


def test_resnet():
    """Test resnet backbone."""
    # Test ResNet12 forward
    model = ResNet12()
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 224, 224)
    feat = model(img)
    assert feat.shape == (1, 640)

    # Test ResNet12 w/o avg pool
    model = ResNet12(with_avgpool=False)
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 224, 224)
    feat = model(img)
    assert feat.shape == (1, 125440)

    # Test ResNet12 w/o flatten
    model = ResNet12(flatten=False)
    model.init_weights()
    model.train()
    img = torch.randn(1, 3, 224, 224)
    feat = model(img)
    assert feat.shape == (1, 640, 1, 1)
