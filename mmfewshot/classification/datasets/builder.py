# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial
from typing import Dict, Optional

from mmcls.datasets import ClassBalancedDataset, ConcatDataset, RepeatDataset
from mmcls.datasets.builder import DATASETS, DistributedSampler, worker_init_fn
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg
from torch.utils.data import DataLoader, Dataset

from mmfewshot.utils import DistributedInfiniteSampler, InfiniteSampler
from mmfewshot.utils import multi_pipeline_collate_fn as collate
from .dataset_wrappers import EpisodicDataset, MetaTestDataset


def build_dataset(cfg: Dict, default_args: Optional[Dict] = None) -> Dataset:
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'EpisodicDataset':
        dataset = EpisodicDataset(
            build_dataset(cfg['dataset'], default_args),
            num_episodes=cfg['num_episodes'],
            num_ways=cfg['num_ways'],
            num_shots=cfg['num_shots'],
            num_queries=cfg['num_queries'])
    elif cfg['type'] == 'MetaTestDataset':
        assert cfg.get('meta_test_cfg', None)
        dataset = MetaTestDataset(
            build_dataset(cfg['dataset'], default_args),
            num_episodes=cfg.meta_test_cfg['num_episodes'],
            num_ways=cfg.meta_test_cfg['num_ways'],
            num_shots=cfg.meta_test_cfg['num_shots'],
            num_queries=cfg.meta_test_cfg['num_queries'])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = True,
                     shuffle: bool = True,
                     round_up: bool = True,
                     seed: Optional[int] = None,
                     pin_memory: bool = False,
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
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default: True.
        seed (int | None): Random seed. Default:None.
        pin_memory (bool): Whether to use pin_memory for dataloader.
            Default: False.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        if use_infinite_sampler:
            sampler = DistributedInfiniteSampler(
                dataset, world_size, rank, shuffle=shuffle)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, round_up=round_up)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = InfiniteSampler(dataset, seed=seed, shuffle=shuffle) \
            if use_infinite_sampler else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle if sampler is None else None,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def build_meta_test_dataloader(dataset: Dataset, meta_test_cfg: Dict,
                               **kwargs) -> DataLoader:
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        meta_test_cfg (dict): Config of meta testing.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        tuple[:obj:`Dataloader`]: `support_data_loader`, `query_data_loader`
            and `test_set_data_loader`.
    """
    support_batch_size = meta_test_cfg.support['batch_size']
    query_batch_size = meta_test_cfg.query['batch_size']
    num_support_workers = meta_test_cfg.support.get('num_workers', 0)
    num_query_workers = meta_test_cfg.query.get('num_workers', 0)

    support_data_loader = DataLoader(
        copy.deepcopy(dataset).support(),
        batch_size=support_batch_size,
        num_workers=num_support_workers,
        collate_fn=partial(collate, samples_per_gpu=support_batch_size),
        pin_memory=False,
        shuffle=True,
        drop_last=meta_test_cfg.support.get('drop_last', False),
        **kwargs)
    query_data_loader = DataLoader(
        copy.deepcopy(dataset).query(),
        batch_size=query_batch_size,
        num_workers=num_query_workers,
        collate_fn=partial(collate, samples_per_gpu=query_batch_size),
        pin_memory=False,
        shuffle=False,
        **kwargs)
    if meta_test_cfg.get('fast_test', False):
        all_batch_size = meta_test_cfg.test_set.get('batch_size', 16)
        num_all_workers = meta_test_cfg.test_set.get('num_workers', 1)
        test_set_data_loader = DataLoader(
            copy.deepcopy(dataset).test_set(),
            batch_size=all_batch_size,
            num_workers=num_all_workers,
            collate_fn=partial(collate, samples_per_gpu=all_batch_size),
            pin_memory=False,
            shuffle=False,
            **kwargs)
    else:
        test_set_data_loader = None
    return support_data_loader, query_data_loader, test_set_data_loader
