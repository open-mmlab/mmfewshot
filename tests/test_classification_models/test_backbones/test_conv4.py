# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.classification.models.backbones.conv4 import (Conv4,
                                                             Conv4NoPool,
                                                             ConvBlock,
                                                             ConvNet)


def test_convblock():

    # ConvBlock w/o pool
    block = ConvBlock(3, 64, False)
    x = torch.randn(1, 3, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 56, 56)

    # ConvBlock w/ pool
    block = ConvBlock(3, 64, True)
    x = torch.randn(1, 3, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)


def test_convnet():
    """Test resnet backbone."""
    # Test ConvNet w/ flatten forward
    model = ConvNet(4, (0, 1, 2, 3), (0, 1, 2, 3), True)
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 1600)

    # Test ConvNet w/o flatten forward
    model = ConvNet(4, (0, 1, 2, 3), (0, 1, 2, 3), False)
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 64, 5, 5)

    # Test Conv4 forward
    model = Conv4()
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 1600)

    # Test Conv4NoPool forward
    model = Conv4NoPool()
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 64, 19, 19)
