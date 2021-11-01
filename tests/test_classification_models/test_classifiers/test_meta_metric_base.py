# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmfewshot.classification.models import CLASSIFIERS


@pytest.mark.parametrize('classifier', [
    'MatchingNetClassifier', 'ProtoNetClassifier', 'RelationNetClassifier',
    'MetaBaselineClassifier'
])
def test_image_classifier(classifier):
    model_cfg = dict(
        type=classifier,
        backbone=(dict(type='Conv4') if classifier != 'RelationNetClassifier'
                  else dict(type='Conv4NoPool')))

    imgs_a = torch.randn(4, 3, 84, 84)
    imgs_b = torch.randn(4, 3, 84, 84)
    label = torch.LongTensor([0, 1, 2, 3])

    model_cfg_ = copy.deepcopy(model_cfg)
    model = CLASSIFIERS.build(model_cfg_)

    # test property
    assert not model.with_neck
    assert model.with_head

    assert model.device
    assert model.get_device()

    # test train_step
    outputs = model.train_step(
        {
            'support_data': {
                'img': imgs_a,
                'gt_label': label,
                'mode': 'train',
                'img_metas': [_ for _ in range(4)]
            },
            'query_data': {
                'img': imgs_b,
                'gt_label': label,
                'mode': 'train',
                'img_metas': [_ for _ in range(4)]
            }
        }, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 4

    # test extract features
    outputs = model(**{
        'img': imgs_a,
        'gt_label': label,
        'mode': 'extract_feat'
    })
    assert outputs.size(0) == 4

    model.before_meta_test(dict())
    model.before_forward_support()

    # test support step
    model(**{'img': imgs_a, 'gt_label': label, 'mode': 'support'})

    model.before_forward_query()
    # test query step
    outputs = model(**{'img': imgs_b, 'gt_label': label, 'mode': 'query'})
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert outputs[0].shape[0] == 4
