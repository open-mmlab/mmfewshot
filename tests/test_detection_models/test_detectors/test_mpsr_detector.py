# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from mmfewshot.detection.models import build_detector
from .test_fine_tune_based_detector import _get_detector_cfg


@pytest.mark.parametrize(
    'cfg_file',
    [
        'detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py',  # noqa
        'detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py'  # noqa
    ])
def test_mpsr_detector_forward(cfg_file):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')
    model = _get_detector_cfg(cfg_file)
    model.backbone.init_cfg = None
    model = build_detector(model)
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    # test train forward
    main_data = dict(
        img_metas=[{
            'img_shape': (256, 256, 3),
            'scale_factor': 1,
            'pad_shape': (256, 256, 3)
        }],
        img=torch.randn(1, 3, 256, 256),
        gt_bboxes=copy.deepcopy(gt_bboxes),
        gt_labels=[torch.LongTensor([0])],
        query_class=torch.LongTensor([0]))
    auxiliary_data = dict()
    scales = [32, 64, 128, 256, 512, 800]
    for i in range(6):
        auxiliary_data.update({
            f'img_metas_scale_{i}': [{
                'img_shape': (scales[i], scales[i], 3),
                'scale_factor': 1,
                'pad_shape': (scales[i], scales[i], 3)
            }],
            f'img_scale_{i}':
            torch.randn(1, 3, scales[i], scales[i]),
            f'gt_bboxes_scale_{i}':
            copy.deepcopy(gt_bboxes),
            f'gt_labels_scale_{i}': [torch.LongTensor([0])]
        })
    model.train()
    losses = model(
        main_data=main_data, auxiliary_data=auxiliary_data, return_loss=True)
    assert 'loss_rpn_cls' in losses
    assert 'loss_rpn_bbox' in losses
    assert 'loss_cls' in losses
    assert 'acc' in losses
    assert 'loss_bbox' in losses
    assert 'loss_rpn_cls_auxiliary' in losses
    assert 'loss_cls_auxiliary' in losses
    assert 'acc_auxiliary' in losses
