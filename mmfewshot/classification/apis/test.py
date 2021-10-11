import copy

import mmcv
import numpy as np
import torch
from mmcls.apis.test import collect_results_cpu
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, get_dist_info
from mmcv.utils import print_log

from mmfewshot.classification.datasets import label_wrapper
from mmfewshot.classification.utils import DeviceWrapper

Z_SCORE = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.98: 2.326,
    0.99: 2.576,
}


def single_gpu_meta_test(model,
                         num_test_tasks,
                         support_dataloader,
                         query_dataloader,
                         test_set_dataloader,
                         meta_test_cfg=None,
                         eval_kwargs=None,
                         logger=None,
                         confidence_interval=0.95,
                         show_task_results=False):
    """Meta testing on single gpu.

    Args:
        model (:obj:`MMDataParallel` | nn.Module): Model to be meta tested.
        num_test_tasks (int): Number of tasks for meta testing.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of query
            data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of all
            test data.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | optional): Logger used for printing
                related information during evaluation. Default: None.
        confidence_interval (float): Confidence interval. Default: 0.95.
        show_task_results (bool): Whether to record the eval result of
            each task. Default: False.

    Returns:
        dict: Dict of meta evaluate results, containing `accuracy_mean`
            and `accuracy_std` of all test tasks.
    """
    assert confidence_interval in Z_SCORE.keys()
    # To avoid deep copying the whole :obj:`MMDataParallel`, we simply
    # copy the module and wrap it with a :class:`DeviceWrapper`.
    # DeviceWrapper will send data to the same device as model.
    if isinstance(model, MMDataParallel):
        model = DeviceWrapper(copy.deepcopy(model.module))
    else:
        model = DeviceWrapper(copy.deepcopy(model))

    if meta_test_cfg.get('fast_test', False):
        print_log('extracting features from all images.', logger=logger)
        extract_features_for_fast_test(model, support_dataloader,
                                       query_dataloader, test_set_dataloader)
    print_log('start meta testing', logger=logger)
    model.before_meta_test(meta_test_cfg)

    results_list = []
    prog_bar = mmcv.ProgressBar(num_test_tasks)
    for task_id in range(num_test_tasks):
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels, logger=logger, **eval_kwargs)
        results_list.append(eval_result)
        prog_bar.update()

    if show_task_results:
        for results in results_list:
            msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
            print_log(msg, logger=logger)

    meta_eval_results = dict()

    for k in results_list[0].keys():
        mean = np.mean([res[k] for res in results_list])
        std = np.std([res[k] for res in results_list])
        std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
        meta_eval_results[f'{k}_mean'] = mean
        meta_eval_results[f'{k}_std'] = std
    return meta_eval_results


def multi_gpu_meta_test(model,
                        num_test_tasks,
                        support_dataloader,
                        query_dataloader,
                        test_set_dataloader=None,
                        meta_test_cfg=None,
                        eval_kwargs=None,
                        logger=None,
                        confidence_interval=0.95,
                        show_task_results=False):
    """Distributed meta testing on multiple gpu, the number of test tasks for
    each GPU is ceil(num_test_tasks / world_size).

    Args:
        model (:obj:`MMDistributedDataParallel`): Model to be meta tested.
        num_test_tasks (int): Number of tasks for meta testing.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            all test data. Default: None.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | optional): Logger used for printing
            related information during evaluation. Default: None.
        confidence_interval (float): Confidence interval. Default: 0.95.
        show_task_results (bool): Whether to record the eval result of
            each task. Default: False.

    Returns:
        dict | None: Dict of meta evaluate results, containing `accuracy_mean`
            and `accuracy_std` of all test tasks.
    """
    assert confidence_interval in Z_SCORE.keys()
    rank, world_size = get_dist_info()
    # Note that each task is tested on a single GPU. Thus the data and model
    # on different GPU should be independent. :obj:`MMDistributedDataParallel`
    # always automatically synchronizes the grad in different GPUs when doing
    # the loss backward, which can not meet the requirements. Thus we simply
    # copy the module and wrap it with an :obj:`DeviceWrapper`, which will
    # send data to the device model.
    model = DeviceWrapper(copy.deepcopy(model.module))
    if meta_test_cfg.get('fast_test', False):
        print_log('extracting features from all images.', logger=logger)
        extract_features_for_fast_test(model, support_dataloader,
                                       query_dataloader, test_set_dataloader)
    print_log('start meta testing', logger=logger)
    model.before_meta_test(meta_test_cfg)

    results_list = []

    sub_num_test_tasks = num_test_tasks // world_size
    sub_num_test_tasks += 1 if num_test_tasks % world_size != 0 else 0
    if rank == 0:
        prog_bar = mmcv.ProgressBar(sub_num_test_tasks)
    for i in range(sub_num_test_tasks):
        task_id = (i * world_size + rank)
        if task_id >= num_test_tasks:
            continue
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels, logger=logger, **eval_kwargs)
        results_list.append(eval_result)
        if rank == 0:
            prog_bar.update()

    collect_results_list = collect_results_cpu(
        results_list, num_test_tasks, tmpdir=None)
    if rank == 0:
        if show_task_results:
            for results in collect_results_list:
                msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
                print_log(msg, logger=logger)

        meta_eval_results = dict()
        print_log(
            f'number of tasks: {len(collect_results_list)}', logger=logger)
        for k in collect_results_list[0].keys():
            mean = np.mean([res[k] for res in collect_results_list])
            std = np.std([res[k] for res in collect_results_list])
            std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
            meta_eval_results[f'{k}_mean'] = mean
            meta_eval_results[f'{k}_std'] = std
        return meta_eval_results
    else:
        return None


def extract_features_for_fast_test(model, support_dataloader, query_dataloader,
                                   test_set_dataloader):
    """Extract features from fixed backbone for all test data to accelerate
    testing. The extracted feats will be saved into `support_dataloader` and
    `query_dataloader`.

    Args:
        model (:obj:`DeviceWrapper`): Model to be meta tested.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            all test data.
    """
    feats_list, img_metas_list = [], []
    rank, _ = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_set_dataloader.dataset))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_set_dataloader):
            img_metas_list.extend(data['img_metas'].data[0])
            feats = model(img=data['img'], mode='extract_feat')
            feats_list.append(feats)
            if rank == 0:
                prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
        feats = torch.cat(feats_list, dim=0)

    query_dataloader.dataset.cache_feats(feats, img_metas_list)
    support_dataloader.dataset.cache_feats(feats, img_metas_list)


def test_single_task(model, support_dataloader, query_dataloader,
                     meta_test_cfg):
    """Task single task.

    Args:
        model (:obj:`DeviceWrapper`): Model to be meta tested.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        meta_test_cfg (dict): Config for meta testing.

    Returns:
        tuple:

            - results_list (list[np.ndarray]): Predict results.
            - gt_labels (np.ndarray): Ground truth labels.
    """
    model = copy.deepcopy(model)
    model.before_forward_support()
    support_cfg = meta_test_cfg.support
    task_class_ids = query_dataloader.dataset.get_task_class_ids()
    if support_cfg.get('train', False):  # methods with fine-tune stage
        optimizer = build_optimizer(model, support_cfg.train['optimizer'])
        num_steps = support_cfg.train['num_steps']
        dataloader_iterator = iter(support_dataloader)
        for i in range(num_steps):
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(support_dataloader)
                data = next(dataloader_iterator)
            data['gt_label'] = label_wrapper(data['gt_label'], task_class_ids)
            optimizer.zero_grad()
            outputs = model.forward(**data, mode='support')
            outputs['loss'].backward()
            optimizer.step()
    else:  # methods without fine-tune stage
        for i, data in enumerate(support_dataloader):
            data['gt_label'] = label_wrapper(data['gt_label'], task_class_ids)
            model.forward(**data, mode='support')

    model.before_forward_query()
    results_list, gt_label_list = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(query_dataloader):
            gt_label_list.append(data.pop('gt_label'))
            result = model.forward(**data, mode='query')
            results_list.extend(result)
        gt_labels = torch.cat(gt_label_list, dim=0).cpu().numpy()
    gt_labels = label_wrapper(gt_labels, task_class_ids)
    return results_list, gt_labels
