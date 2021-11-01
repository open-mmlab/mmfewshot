# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmfewshot.classification.models.backbones.wrn import (WideResNet,
                                                           WRN28x10, WRNBlock)


def test_wrnblock():
    # WRNBlock stride 1
    block = WRNBlock(64, 64, 1, 0.0)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 56, 56)

    # WRNBlock stride 2
    block = WRNBlock(64, 64, 2, 0.0)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)

    # WRNBlock stride 2 w/ drop out
    block = WRNBlock(64, 64, 2, 0.1)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 28, 28)


def test_wide_resnet():
    """Test resnet backbone."""
    # Test WRN forward
    model = WideResNet(28, 10, 1, 0.5, True, True, (1, 1))
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 640)

    # Test WRN w/o avg pool
    model = WideResNet(28, 10, 1, 0.5, True, False, (1, 1))
    model.init_weights()
    model.train()

    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 282240)

    # Test WRN w/o flatten and avgpool
    model = WideResNet(28, 10, 1, 0.5, False, False, (1, 1))
    model.init_weights()
    model.train()
    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 640, 21, 21)

    # Test WRN28x10
    model = WRN28x10()
    model.init_weights()
    model.train()
    img = torch.randn(1, 3, 84, 84)
    feat = model(img)
    assert feat.shape == (1, 640)
