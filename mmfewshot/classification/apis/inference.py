# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcls.core.visualization import imshow_infos
from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmfewshot.classification.models import BaseMetricClassifier


def init_classifier(config: Union[str, mmcv.Config],
                    checkpoint: Optional[str] = None,
                    device: str = 'cuda:0',
                    options: Optional[Dict] = None) -> nn.Module:
    """Prepare a few shot classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str | None): Checkpoint path. If left as None, the model
            will not load any weights. Default: None.
        device (str): Runtime device. Default: 'cuda:0'.
        options (dict | None): Options to override some settings in the
            used config. Default: None.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    model = build_classifier(config.model)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        load_checkpoint(model, checkpoint, map_location=map_loc)
    # save the config in the model for convenience in later use
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def process_support_images(model: nn.Module, support_imgs: List[str],
                           support_labels: List[str]) -> None:
    """Process support images.

    Args:
        model (nn.Module): Classifier model.
        support_imgs (list[str]): The image filenames.
        support_labels (list[str]): The class names of support images.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    pipeline = cfg.data.test.dataset.pipeline
    if pipeline[0]['type'] != 'LoadImageFromFile':
        pipeline[0]['type'] = 'LoadImageFromFile'
    test_pipeline = Compose(pipeline)
    model.CLASSES = list(set(support_labels))
    cat_to_id = {cat: i for i, cat in enumerate(model.CLASSES)}
    model.before_forward_support()
    # forward support images
    with torch.no_grad():
        for img, label in zip(support_imgs, support_labels):
            data = dict(
                img_info=dict(filename=img),
                gt_label=np.array(cat_to_id[label], dtype=np.int64),
                img_prefix=None)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
                model(mode='support', **data)
    model.before_forward_query()


def inference_classifier(model: nn.Module, query_img: str) -> Dict:
    """Inference single image with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        query_img (str): The image filename.

    Returns:
        dict: The classification results that contains
            `pred_score` of each class.
    """
    # only support methods without fine-tuning
    if isinstance(model, BaseMetricClassifier):
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # build the data pipeline
        pipeline = cfg.data.test.dataset.pipeline
        if pipeline[0]['type'] != 'LoadImageFromFile':
            pipeline[0]['type'] = 'LoadImageFromFile'
        test_pipeline = Compose(pipeline)
        data = dict(
            img_info=dict(filename=query_img),
            gt_label=np.array(-1, dtype=np.int64),
            img_prefix=None)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]

        # inference image
        with torch.no_grad():
            scores = model(mode='query', img=data['img'])[0]
            result = {
                model.CLASSES[i]: float(scores[i])
                for i in range(scores.shape[0])
            }
        return result
    else:
        raise TypeError(
            'currently, inference only support metric based methods')


def show_result_pyplot(img: str,
                       result: Dict,
                       fig_size: Tuple[int] = (15, 10),
                       wait_time: int = 0,
                       out_file: Optional[str] = None) -> np.ndarray:
    """Visualize the classification results on the image.

    Args:
        img (str): Image filename.
        result (dict): The classification result.
        fig_size (tuple): Figure size of the pyplot figure. Default: (15, 10).
        wait_time (int): How many seconds to display the image. Default: 0.
        out_file (str | None): Default: None

    Returns:
        np.ndarray: pyplot figure.
    """
    img = mmcv.imread(img)
    img = img.copy()
    img = imshow_infos(
        img,
        result,
        text_color='white',
        font_size=25,
        row_width=20,
        win_name='',
        show=True,
        fig_size=fig_size,
        wait_time=wait_time,
        out_file=out_file)
    return img
