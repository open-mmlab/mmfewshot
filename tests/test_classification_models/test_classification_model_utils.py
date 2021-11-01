# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmfewshot.classification.models.backbones import Conv4
from mmfewshot.classification.models.utils import convert_maml_module


def test_maml_module():
    model = Conv4()
    maml_model = convert_maml_module(model)
    image = torch.randn(1, 3, 32, 32)
    for weight in maml_model.parameters():
        assert weight.fast is None
    feat = maml_model(image)
    for weight in maml_model.parameters():
        weight.fast = weight
    maml_feat = maml_model(image)
    assert torch.allclose(feat, maml_feat)
