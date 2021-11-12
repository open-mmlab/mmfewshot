# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from mmfewshot.detection.models import QuerySupportDetector


def init_detector(config: Union[str, mmcv.Config],
                  checkpoint: Optional[str] = None,
                  device: str = 'cuda:0',
                  cfg_options: Optional[Dict] = None,
                  classes: Optional[List[str]] = None) -> nn.Module:
    """Prepare a detector from config file.

    Args:
        config (str | :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str | None): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Runtime device. Default: 'cuda:0'.
        cfg_options (dict | None): Options to override some settings in the
            used config.
        classes (list[str] | None): Options to override classes name of model.
            Default: None.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        if classes is not None:
            model.CLASSES = classes
    # save the config in the model for convenience in later use
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def process_support_images(model: nn.Module,
                           support_imgs: List[str],
                           support_labels: List[List[str]],
                           support_bboxes: Optional[List[List[float]]] = None,
                           classes: Optional[List[str]] = None) -> None:
    """Process support images for query support detector.

    Args:
        model (nn.Module): Detector.
        support_imgs (list[str]): Support image filenames.
        support_labels (list[list[str]]): Support labels of each bbox.
        support_bboxes (list[list[list[float]]] | None): Bbox in support
            images. If it set to None, it will use the [0, 0, image width,
            image height] as bbox. Default: None.
        classes (list[str] | None): Options to override classes name of model.
            Default: None.
    """
    if isinstance(model, QuerySupportDetector):
        cfg = model.cfg
        # build pipeline
        support_pipeline = cfg.data.model_init.pipeline
        support_pipeline = replace_ImageToTensor(support_pipeline)
        support_pipeline = Compose(support_pipeline)
        # update classes
        if classes is not None:
            model.CLASSES = classes
        cat_to_id = {cat: i for i, cat in enumerate(model.CLASSES)}
        if isinstance(support_imgs, str):
            support_imgs = [support_imgs]
        device = next(model.parameters()).device  # model device
        assert len(support_imgs) == len(support_labels)
        data_list = []
        for i, img in enumerate(support_imgs):
            if support_bboxes is None:
                img_shape = mmcv.imread(img).shape
                bboxes = [[0, 0, img_shape[1], img_shape[0]]]
            else:
                bboxes = support_bboxes[i]
            labels = [cat_to_id[label] for label in support_labels[i]]
            ann_info = dict(
                bboxes=np.array(bboxes, dtype=np.float32),
                labels=np.array(labels),
                dtype=np.int64)
            # prepare data
            data = dict(
                img_info=dict(filename=img, ann=ann_info),
                ann_info=ann_info,
                bbox_fields=[],
                img_prefix=None)
            # build the data pipeline
            data = support_pipeline(data)
            data_list.append(data)

        data = collate(data_list, samples_per_gpu=len(support_imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas for img_metas in data['img_metas'].data[0]
        ]
        data['img'] = data['img'].data[0]
        data['gt_bboxes'] = [img for img in data['gt_bboxes'].data[0]]
        data['gt_labels'] = [img for img in data['gt_labels'].data[0]]
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # initialize model
        with torch.no_grad():
            model(mode='model_init', **data)
            model.model_init()
    else:
        raise TypeError('currently, only support query support detector.')


def inference_detector(model: nn.Module, imgs: Union[List[str], str]) -> List:
    """Inference images with the detector.

    Args:
        model (nn.Module): Detector.
        imgs (list[str] | str): Batch or single image file.

    Returns:
        list: If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data_list = []
    for img in imgs:
        # prepare data
        data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        data_list.append(data)

    data = collate(data_list, samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        if isinstance(model, QuerySupportDetector):
            results = model(mode='test', rescale=True, **data)
        else:
            results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
