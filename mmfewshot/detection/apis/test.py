import os.path as osp
import time

import mmcv
import torch
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu, collect_results_gpu
from mmdet.core import encode_mask_results
from mmdet.utils import get_root_logger


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu for meta-learning based detector.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_dir (str or None): The dir to write the image.
            Default: None.
        show_score_thr (float, optional): Minimum score of bboxes to be shown.
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

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus for meta-learning based detector.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

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
            result = model(mode='test', rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_model_init(model, data_loader):
    """Extracting support template features for meta-learning methods in query-
    support fashion with single gpu.

    Args:
        model (nn.Module): Model used for extract support template features.
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
            result = model(mode='model_init', **data)
        results.append(result)
        for _ in range(len(data['img_metas'].data[0])):
            prog_bar.update()
    model.module.model_init()
    logger.info('model initialization done.')

    return results


def multi_gpu_model_init(model, data_loader):
    """Extracting support template features for meta-learning methods in query-
    support fashion with multi gpus.

    Args:
        model (nn.Module): Model used for extract support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        logger = get_root_logger()
        logger.info('starting model initialization...')
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(mode='model_init', **data)
        results.append(result)
        if rank == 0:
            for _ in range(len(data['img_metas'].data[0])):
                prog_bar.update()
    model.module.model_init()
    if rank == 0:
        logger.info('model initialization done.')
    return results
