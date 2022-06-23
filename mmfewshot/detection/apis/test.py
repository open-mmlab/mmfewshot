# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import time
from typing import List, Optional

import mmcv
import torch
import torch.nn as nn
from mmcv.image import tensor2imgs
from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu, collect_results_gpu
from mmdet.utils import get_root_logger
from torch.utils.data import DataLoader


def single_gpu_test(model: nn.Module,
                    data_loader: DataLoader,
                    show: bool = False,
                    out_dir: Optional[str] = None,
                    show_score_thr: float = 0.3) -> List:
    """Test model with single gpu for meta-learning based detector.

    The model forward function requires `mode`, while in mmdet it requires
    `return_loss`. And the `encode_mask_results` is removed.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (DataLoader): Pytorch data loader.
        show (bool): Whether to show the image. Default: False.
        out_dir (str | None): The directory to write the image. Default: None.
        show_score_thr (float): Minimum score of bboxes to be shown.
            Default: 0.3.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `test` mode
            result = model(mode='test', rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            # make sure each time only one image to be shown
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                if is_module_wrapper(model):
                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
                else:
                    model.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        results.extend(result)

        prog_bar.update(batch_size)
    return results


def multi_gpu_test(model: nn.Module,
                   data_loader: DataLoader,
                   tmpdir: str = None,
                   gpu_collect: bool = False) -> List:
    """Test model with multiple gpus for meta-learning based detector.

    The model forward function requires `mode`, while in mmdet it requires
    `return_loss`. And the `encode_mask_results` is removed.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `test` mode
            result = model(mode='test', rescale=True, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            prog_bar.update(batch_size * world_size)

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_model_init(model: nn.Module, data_loader: DataLoader) -> List:
    """Forward support images for meta-learning based detector initialization.

    The function usually will be called before `single_gpu_test` in
    `QuerySupportEvalHook`. It firstly forwards support images with
    `mode=model_init` and the features will be saved in the model.
    Then it will call `:func:model_init` to process the extracted features
    of support images to finish the model initialization.

    Args:
        model (nn.Module): Model used for extracting support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    logger = get_root_logger()
    logger.info('starting model initialization...')
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `model_init` mode
            result = model(mode='model_init', **data)
        results.append(result)
        prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    # `model_init` will process the forward features saved in model.
    if is_module_wrapper(model):
        model.module.model_init()
    else:
        model.model_init()
    logger.info('model initialization done.')

    return results


def multi_gpu_model_init(model: nn.Module, data_loader: DataLoader) -> List:
    """Forward support images for meta-learning based detector initialization.

    The function usually will be called before `single_gpu_test` in
    `QuerySupportEvalHook`. It firstly forwards support images with
    `mode=model_init` and the features will be saved in the model.
    Then it will call `:func:model_init` to process the extracted features
    of support images to finish the model initialization.

    Noted that the `data_loader` should NOT use distributed sampler, all the
    models in different gpus should be initialized with same images.

    Args:
        model (nn.Module): Model used for extracting support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, _ = get_dist_info()
    if rank == 0:
        logger = get_root_logger()
        logger.info('starting model initialization...')
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    # the model_init dataloader do not use distributed sampler to make sure
    # all of the gpus get the same initialization
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `model_init` mode
            result = model(mode='model_init', **data)
        results.append(result)
        if rank == 0:
            prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    # model_init function will process the forward features saved in model.
    if is_module_wrapper(model):
        model.module.model_init()
    else:
        model.model_init()
    if rank == 0:
        logger.info('model initialization done.')
    return results
