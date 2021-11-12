# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmfewshot.classification.models import CLASSIFIERS
from mmfewshot.classification.utils import MetaTestParallel


@pytest.mark.parametrize('classifier',
                         ['Baseline', 'BaselinePlus', 'NegMargin'])
def test_image_classifier(classifier):
    model_cfg = dict(type=classifier, backbone=dict(type='Conv4'))

    imgs = torch.randn(4, 3, 84, 84)
    feats = torch.randn(4, 1600)
    label = torch.LongTensor([0, 1, 2, 3])

    model_cfg_ = copy.deepcopy(model_cfg)
    model = CLASSIFIERS.build(model_cfg_)

    # test property
    assert not model.with_neck
    assert model.with_head

    assert model.device
    assert model.get_device()

    model = MetaTestParallel(copy.deepcopy(model))

    # test extract features
    outputs = model(**{'img': imgs, 'gt_label': label, 'mode': 'extract_feat'})
    assert outputs.size(0) == 4

    model.before_meta_test(dict())
    model.before_forward_support()

    # test support step
    outputs = model(**{'img': imgs, 'gt_label': label, 'mode': 'support'})
    assert outputs['loss'].item() > 0

    # test support step
    outputs = model(**{'feats': feats, 'gt_label': label, 'mode': 'support'})
    assert outputs['loss'].item() > 0

    model.before_forward_query()
    # test query step
    outputs = model(**{'img': imgs, 'mode': 'query'})
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert outputs[0].shape[0] == 5

    # test query step
    outputs = model(**{'feats': feats, 'mode': 'query'})
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert outputs[0].shape[0] == 5

    with pytest.raises(ValueError):
        # test extract features
        outputs = model(**{'img': imgs, 'gt_label': label, 'mode': 'test'})
