# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from functools import partial
from typing import Dict, Optional, Tuple

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import ConfigDict, build_from_cfg
from mmdet.datasets.builder import DATASETS, worker_init_fn
from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                             ConcatDataset, RepeatDataset)
from mmdet.datasets.samplers import (DistributedGroupSampler,
                                     DistributedSampler, GroupSampler)
from torch.utils.data import DataLoader, Dataset, Sampler

from mmfewshot.utils.infinite_sampler import (DistributedInfiniteGroupSampler,
                                              DistributedInfiniteSampler,
                                              InfiniteGroupSampler)
from .dataset_wrappers import (NWayKShotDataset, QueryAwareDataset,
                               TwoBranchDataset)
from .utils import get_copy_dataset_type


def build_dataset(cfg: ConfigDict,
                  default_args: Dict = None,
                  rank: Optional[int] = None,
                  work_dir: Optional[str] = None,
                  timestamp: Optional[str] = None) -> Dataset:
    # If save_dataset is set to True, dataset will be saved into json.
    save_dataset = cfg.pop('save_dataset', False)

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'QueryAwareDataset':
        query_dataset = build_dataset(cfg['dataset'], default_args)
        if cfg.get('support_dataset', None) is not None:
            # if `copy_from_query_dataset` is True, copy and update config
            # from query_dataset and copy `data_infos` by using copy dataset
            # to avoid reproducing random sampling.
            if cfg['support_dataset'].pop('copy_from_query_dataset', False):
                support_dataset_cfg = copy.deepcopy(cfg['dataset'])
                support_dataset_cfg.update(cfg['support_dataset'])
                support_dataset_cfg['type'] = get_copy_dataset_type(
                    cfg['dataset']['type'])
                support_dataset_cfg['ann_cfg'] = [
                    dict(data_infos=copy.deepcopy(query_dataset.data_infos))
                ]
                cfg['support_dataset'] = support_dataset_cfg
            support_dataset = build_dataset(cfg['support_dataset'],
                                            default_args)
        else:
            support_dataset = None

        dataset = QueryAwareDataset(
            query_dataset,
            support_dataset,
            num_support_ways=cfg['num_support_ways'],
            num_support_shots=cfg['num_support_shots'],
            repeat_times=cfg.get('repeat_times', 1))
    elif cfg['type'] == 'NWayKShotDataset':
        query_dataset = build_dataset(cfg['dataset'], default_args)
        if cfg.get('support_dataset', None) is not None:
            # if `copy_from_query_dataset` is True, copy and update config
            # from query_dataset and copy `data_infos` by using copy dataset
            # to avoid reproducing random sampling.
            if cfg['support_dataset'].pop('copy_from_query_dataset', False):
                support_dataset_cfg = copy.deepcopy(cfg['dataset'])
                support_dataset_cfg.update(cfg['support_dataset'])
                support_dataset_cfg['type'] = get_copy_dataset_type(
                    cfg['dataset']['type'])
                support_dataset_cfg['ann_cfg'] = [
                    dict(data_infos=copy.deepcopy(query_dataset.data_infos))
                ]
                cfg['support_dataset'] = support_dataset_cfg
            support_dataset = build_dataset(cfg['support_dataset'],
                                            default_args)
        else:
            support_dataset = None

        dataset = NWayKShotDataset(
            query_dataset,
            support_dataset,
            num_support_ways=cfg['num_support_ways'],
            num_support_shots=cfg['num_support_shots'],
            one_support_shot_per_image=cfg.get('one_support_shot_per_image',
                                               False),
            num_used_support_shots=cfg.get('num_used_support_shots', None),
            shuffle_support=cfg.get('shuffle_support', False),
            repeat_times=cfg.get('repeat_times', 1),
        )
    elif cfg['type'] == 'TwoBranchDataset':
        main_dataset = build_dataset(cfg['dataset'], default_args)
        # if `copy_from_main_dataset` is True, copy and update config
        # from main_dataset and copy `data_infos` by using copy dataset
        # to avoid reproducing random sampling.
        if cfg['auxiliary_dataset'].pop('copy_from_main_dataset', False):
            auxiliary_dataset_cfg = copy.deepcopy(cfg['dataset'])
            auxiliary_dataset_cfg.update(cfg['auxiliary_dataset'])
            auxiliary_dataset_cfg['type'] = get_copy_dataset_type(
                cfg['dataset']['type'])
            auxiliary_dataset_cfg['ann_cfg'] = [
                dict(data_infos=copy.deepcopy(main_dataset.data_infos))
            ]
            cfg['auxiliary_dataset'] = auxiliary_dataset_cfg
        auxiliary_dataset = build_dataset(cfg['auxiliary_dataset'],
                                          default_args)
        dataset = TwoBranchDataset(
            main_dataset=main_dataset,
            auxiliary_dataset=auxiliary_dataset,
            reweight_dataset=cfg.get('reweight_dataset', False))
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    # save dataset for the reproducibility
    if rank == 0 and save_dataset:
        save_dataset_path = osp.join(work_dir, f'{timestamp}_saved_data.json')
        if hasattr(dataset, 'save_data_infos'):
            dataset.save_data_infos(save_dataset_path)
        else:
            raise AttributeError(
                f'`save_data_infos` is not implemented in {type(dataset)}.')

    return dataset


def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = True,
                     shuffle: bool = True,
                     seed: Optional[int] = None,
                     data_cfg: Optional[Dict] = None,
                     use_infinite_sampler: bool = False,
                     **kwargs) -> DataLoader:
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
            Default:1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int): Random seed. Default:None.
        data_cfg (dict | None): Dict of data configure. Default: None.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    (sampler, batch_size, num_workers) = build_sampler(
        dist=dist,
        shuffle=shuffle,
        dataset=dataset,
        num_gpus=num_gpus,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        seed=seed,
        use_infinite_sampler=use_infinite_sampler)
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    if isinstance(dataset, QueryAwareDataset):
        from mmfewshot.utils import multi_pipeline_collate_fn
        # `QueryAwareDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
    elif isinstance(dataset, NWayKShotDataset):
        from .dataloader_wrappers import NWayKShotDataloader
        from mmfewshot.utils import multi_pipeline_collate_fn
        # `NWayKShotDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]

        # initialize query dataloader
        query_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # creat support dataset from query dataset and
        # sample batch index with same length as query dataloader
        support_dataset = copy.deepcopy(dataset)
        support_dataset.convert_query_to_support(
            len(query_data_loader) * num_gpus)

        (support_sampler, _, _) = build_sampler(
            dist=dist,
            shuffle=False,
            dataset=support_dataset,
            num_gpus=num_gpus,
            samples_per_gpu=1,
            workers_per_gpu=workers_per_gpu,
            seed=seed,
            use_infinite_sampler=use_infinite_sampler)

        # wrap two dataloaders with dataloader wrapper
        data_loader = NWayKShotDataloader(
            query_data_loader=query_data_loader,
            support_dataset=support_dataset,
            support_sampler=support_sampler,
            num_workers=num_workers,
            support_collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=1),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
    elif isinstance(dataset, TwoBranchDataset):
        from .dataloader_wrappers import TwoBranchDataloader
        from mmfewshot.utils import multi_pipeline_collate_fn
        # `TwoBranchDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]

        # initialize main dataloader
        main_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # convert main dataset to auxiliary dataset
        auxiliary_dataset = copy.deepcopy(dataset)
        auxiliary_dataset.convert_main_to_auxiliary()
        # initialize auxiliary sampler and dataloader
        auxiliary_samples_per_gpu = \
            data_cfg.get('auxiliary_samples_per_gpu', samples_per_gpu)
        auxiliary_workers_per_gpu = \
            data_cfg.get('auxiliary_workers_per_gpu', workers_per_gpu)
        (auxiliary_sampler, auxiliary_batch_size,
         auxiliary_num_workers) = build_sampler(
             dist=dist,
             shuffle=shuffle,
             dataset=auxiliary_dataset,
             num_gpus=num_gpus,
             samples_per_gpu=auxiliary_samples_per_gpu,
             workers_per_gpu=auxiliary_workers_per_gpu,
             seed=seed,
             use_infinite_sampler=use_infinite_sampler)
        auxiliary_data_loader = DataLoader(
            auxiliary_dataset,
            batch_size=auxiliary_batch_size,
            sampler=auxiliary_sampler,
            num_workers=auxiliary_num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn,
                samples_per_gpu=auxiliary_samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # wrap two dataloaders with dataloader wrapper
        data_loader = TwoBranchDataloader(
            main_data_loader=main_data_loader,
            auxiliary_data_loader=auxiliary_data_loader,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)

    return data_loader


def build_sampler(
        dist: bool,
        shuffle: bool,
        dataset: Dataset,
        num_gpus: int,
        samples_per_gpu: int,
        workers_per_gpu: int,
        seed: int,
        use_infinite_sampler: bool = False) -> Tuple[Sampler, int, int]:
    """Build pytorch sampler for dataLoader.

    Args:
        dist (bool): Distributed training/test or not.
        shuffle (bool): Whether to shuffle the data at every epoch.
        dataset (Dataset): A PyTorch dataset.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        seed (int): Random seed.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.

    Returns:
        tuple: Contains corresponding sampler and arguments

            - sampler(:obj:`sampler`) : Corresponding sampler
              used in dataloader.
            - batch_size(int): Batch size of dataloader.
            - num_works(int): The number of processes loading data in the
                data loader.
    """

    rank, world_size = get_dist_info()
    if dist:
        # Infinite sampler will return a infinite stream of index. But,
        # the length of infinite sampler is set to the actual length of
        # dataset, thus the length of dataloader is still determined
        # by the dataset.

        if shuffle:
            if use_infinite_sampler:
                sampler = DistributedInfiniteGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                # DistributedGroupSampler will definitely shuffle the data to
                # satisfy that images on each GPU are in the same group
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            if use_infinite_sampler:
                sampler = DistributedInfiniteSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if use_infinite_sampler:
            sampler = InfiniteGroupSampler(
                dataset, samples_per_gpu, seed=seed, shuffle=shuffle)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) \
                if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    return sampler, batch_size, num_workers
