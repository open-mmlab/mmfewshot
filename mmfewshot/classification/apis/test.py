# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Union

import mmcv
import numpy as np
import torch
from mmcls.apis.test import collect_results_cpu
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, get_dist_info
from mmcv.utils import print_log
from torch import nn
from torch.utils.data import DataLoader

from mmfewshot.classification.datasets import label_wrapper
from mmfewshot.classification.utils import MetaTestParallel

# z scores of different confidence intervals
Z_SCORE = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.98: 2.326,
    0.99: 2.576,
}


def single_gpu_meta_test(model: Union[MMDataParallel, nn.Module],
                         num_test_tasks: int,
                         support_dataloader: DataLoader,
                         query_dataloader: DataLoader,
                         test_set_dataloader: Optional[DataLoader] = None,
                         meta_test_cfg: Optional[Dict] = None,
                         eval_kwargs: Optional[Dict] = None,
                         logger: Optional[object] = None,
                         confidence_interval: float = 0.95,
                         show_task_results: bool = False) -> Dict:
    """Meta testing on single gpu.

    During meta testing, model might be further fine-tuned or added extra
    parameters. While the tested model need to be restored after meta
    testing since meta testing can be used as the validation in the middle
    of training. To detach model from previous phase, the model will be
    copied and wrapped with :obj:`MetaTestParallel`. And it has full
    independence from the training model and will be discarded after the
    meta testing.

    Args:
        model (:obj:`MMDataParallel` | nn.Module): Model to be meta tested.
        num_test_tasks (int): Number of meta testing tasks.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data and it is used to fetch support data for each task.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of query
            data and it is used to fetch query data for each task.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of all
            test data and it is used for feature extraction from whole dataset
            to accelerate the testing. Default: None.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | None): Logger used for printing
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
    # copy the module and wrap it with a :class:`MetaTestParallel`.
    # MetaTestParallel will send data to the same device as model.
    if isinstance(model, MMDataParallel):
        model = MetaTestParallel(copy.deepcopy(model.module))
    else:
        model = MetaTestParallel(copy.deepcopy(model))

    # for the backbone-fixed methods, the features can be pre-computed
    # and saved in dataset to achieve acceleration
    if meta_test_cfg.get('fast_test', False):
        print_log('extracting features from all images.', logger=logger)
        extract_features_for_fast_test(model, support_dataloader,
                                       query_dataloader, test_set_dataloader)
    print_log('start meta testing', logger=logger)

    # prepare for meta test
    model.before_meta_test(meta_test_cfg)

    results_list = []
    prog_bar = mmcv.ProgressBar(num_test_tasks)
    for task_id in range(num_test_tasks):
        # set support and query dataloader to the same task by task id
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        # test a task
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        # evaluate predict result
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels, logger=logger, **eval_kwargs)
        eval_result['task_id'] = task_id
        results_list.append(eval_result)
        prog_bar.update()

    if show_task_results:
        # the result of each task will be logged into logger
        for results in results_list:
            msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
            print_log(msg, logger=logger)

    meta_eval_results = dict()
    # get the average accuracy and std
    for k in results_list[0].keys():
        if k == 'task_id':
            continue
        mean = np.mean([res[k] for res in results_list])
        std = np.std([res[k] for res in results_list])
        std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
        meta_eval_results[f'{k}_mean'] = mean
        meta_eval_results[f'{k}_std'] = std
    return meta_eval_results


def multi_gpu_meta_test(model: MMDistributedDataParallel,
                        num_test_tasks: int,
                        support_dataloader: DataLoader,
                        query_dataloader: DataLoader,
                        test_set_dataloader: Optional[DataLoader] = None,
                        meta_test_cfg: Optional[Dict] = None,
                        eval_kwargs: Optional[Dict] = None,
                        logger: Optional[object] = None,
                        confidence_interval: float = 0.95,
                        show_task_results: bool = False) -> Dict:
    """Distributed meta testing on multiple gpus.

    During meta testing, model might be further fine-tuned or added extra
    parameters. While the tested model need to be restored after meta
    testing since meta testing can be used as the validation in the middle
    of training. To detach model from previous phase, the model will be
    copied and wrapped with :obj:`MetaTestParallel`. And it has full
    independence from the training model and will be discarded after the
    meta testing.

    In the distributed situation, the :obj:`MetaTestParallel` on each GPU
    is also independent. The test tasks in few shot leaning usually are very
    small and hardly benefit from distributed acceleration. Thus, in
    distributed meta testing, each task is done in single GPU and each GPU
    is assigned a certain number of tasks. The number of test tasks
    for each GPU is ceil(num_test_tasks / world_size). After all GPUs finish
    their tasks, the results will be aggregated to get the final result.

    Args:
        model (:obj:`MMDistributedDataParallel`): Model to be meta tested.
        num_test_tasks (int): Number of meta testing tasks.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            all test data. Default: None.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | None): Logger used for printing
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
    # copy the module and wrap it with an :obj:`MetaTestParallel`, which will
    # send data to the device model.
    model = MetaTestParallel(copy.deepcopy(model.module))

    # for the backbone-fixed methods, the features can be pre-computed
    # and saved in dataset to achieve acceleration
    if meta_test_cfg.get('fast_test', False):
        print_log('extracting features from all images.', logger=logger)
        extract_features_for_fast_test(model, support_dataloader,
                                       query_dataloader, test_set_dataloader)
    print_log('start meta testing', logger=logger)
    # prepare for meta test
    model.before_meta_test(meta_test_cfg)

    results_list = []

    # tasks will be evenly distributed on each gpus
    sub_num_test_tasks = num_test_tasks // world_size
    sub_num_test_tasks += 1 if num_test_tasks % world_size != 0 else 0
    if rank == 0:
        prog_bar = mmcv.ProgressBar(num_test_tasks)
    for i in range(sub_num_test_tasks):
        task_id = (i * world_size + rank)
        if task_id >= num_test_tasks:
            continue
        # set support and query dataloader to the same task by task id
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        # test a task
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        # evaluate predict result
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels, logger=logger, **eval_kwargs)
        eval_result['task_id'] = task_id
        results_list.append(eval_result)
        if rank == 0:
            prog_bar.update(world_size)

    collect_results_list = collect_results_cpu(
        results_list, num_test_tasks, tmpdir=None)
    if rank == 0:
        if show_task_results:
            # the result of each task will be logged into logger
            for results in collect_results_list:
                msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
                print_log(msg, logger=logger)

        meta_eval_results = dict()
        print_log(
            f'number of tasks: {len(collect_results_list)}', logger=logger)
        # get the average accuracy and std
        for k in collect_results_list[0].keys():
            if k == 'task_id':
                continue
            mean = np.mean([res[k] for res in collect_results_list])
            std = np.std([res[k] for res in collect_results_list])
            std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
            meta_eval_results[f'{k}_mean'] = mean
            meta_eval_results[f'{k}_std'] = std
        return meta_eval_results
    else:
        return None


def extract_features_for_fast_test(model: MetaTestParallel,
                                   support_dataloader: DataLoader,
                                   query_dataloader: DataLoader,
                                   test_set_dataloader: DataLoader) -> None:
    """Extracting and saving features for testing acceleration.

    In some methods, the backbone is fixed during meta testing, which results
    in the features from backbone are also fixed for whole dataset. So we can
    calculate the features in advance and save them into `support_dataloader`
    and `query_dataloader`. In this way, the model can skip the feature
    extraction phase during the meta testing, which can obviously accelerate
    the meta testing.

    Args:
        model (:obj:`MetaTestParallel`): Model to be meta tested.
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
    # traverse the whole dataset and compute the features from backbone
    with torch.no_grad():
        for i, data in enumerate(test_set_dataloader):
            img_metas_list.extend(data['img_metas'].data[0])
            # forward in `extract_feat` mode
            feats = model(img=data['img'], mode='extract_feat')
            feats_list.append(feats)
            if rank == 0:
                prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
        feats = torch.cat(feats_list, dim=0)
    # cache the pre-computed features into dataset
    query_dataloader.dataset.cache_feats(feats, img_metas_list)
    support_dataloader.dataset.cache_feats(feats, img_metas_list)


def test_single_task(model: MetaTestParallel, support_dataloader: DataLoader,
                     query_dataloader: DataLoader, meta_test_cfg: Dict):
    """Test a single task.

    A task has two stages: handling the support set and predicting the
    query set. In stage one, it currently supports fine-tune based and
    metric based methods. In stage two, it simply forward the query set
    and gather all the results.

    Args:
        model (:obj:`MetaTestParallel`): Model to be meta tested.
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
    # use copy of model for each task
    model = copy.deepcopy(model)
    # get ids of all classes in this task
    task_class_ids = query_dataloader.dataset.get_task_class_ids()

    # forward support set
    model.before_forward_support()
    support_cfg = meta_test_cfg.get('support', dict())
    # methods with fine-tune stage
    if support_cfg.get('train', False):
        optimizer = build_optimizer(model, support_cfg.train['optimizer'])
        num_steps = support_cfg.train['num_steps']
        dataloader_iterator = iter(support_dataloader)
        for i in range(num_steps):
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(support_dataloader)
                data = next(dataloader_iterator)
            # map input labels into range of 0 to numbers of classes-1
            data['gt_label'] = label_wrapper(data['gt_label'], task_class_ids)
            optimizer.zero_grad()
            # forward in `support` mode
            outputs = model.forward(**data, mode='support')
            outputs['loss'].backward()
            optimizer.step()
    # methods without fine-tune stage
    else:
        for i, data in enumerate(support_dataloader):
            # map input labels into range of 0 to numbers of classes-1
            data['gt_label'] = label_wrapper(data['gt_label'], task_class_ids)
            # forward in `support` mode
            model.forward(**data, mode='support')

    # forward query set
    model.before_forward_query()
    results_list, gt_label_list = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(query_dataloader):
            gt_label_list.append(data.pop('gt_label'))
            # forward in `query` mode
            result = model.forward(**data, mode='query')
            results_list.extend(result)
        gt_labels = torch.cat(gt_label_list, dim=0).cpu().numpy()
    # map gt labels into range of 0 to numbers of classes-1.
    gt_labels = label_wrapper(gt_labels, task_class_ids)
    return results_list, gt_labels
