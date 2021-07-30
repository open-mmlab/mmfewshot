import copy
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import DATASETS, worker_init_fn
from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                             ConcatDataset, RepeatDataset)
from mmdet.datasets.samplers import (DistributedGroupSampler,
                                     DistributedSampler, GroupSampler)
from torch.utils.data import DataLoader

from .dataset_wrappers import NwayKshotDataset, QueryAwareDataset


def build_dataset(cfg, default_args=None):
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
        dataset = QueryAwareDataset(
            query_dataset=build_dataset(cfg['dataset'], default_args),
            support_dataset=build_dataset(cfg['support_dataset'], default_args)
            if cfg.get('support_dataset', None) is not None else None,
            num_support_ways=cfg['num_support_ways'],
            num_support_shots=cfg['num_support_shots'],
            repeat_times=cfg.get('repeat_times', 1))
    elif cfg['type'] == 'NwayKshotDataset':
        dataset = NwayKshotDataset(
            query_dataset=build_dataset(cfg['dataset'], default_args),
            support_dataset=build_dataset(cfg['support_dataset'], default_args)
            if cfg.get('support_dataset', None) is not None else None,
            num_support_ways=cfg['num_support_ways'],
            num_support_shots=cfg['num_support_shots'],
            mutual_support_shot=cfg.get('mutual_support_shot', False),
            num_used_support_shots=cfg.get('num_used_support_shots', None),
            shuffle_support=cfg.get('use_shuffle_support', False),
            repeat_times=cfg.get('repeat_times', 1),
        )
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
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
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    (sampler, batch_size, num_workers) \
        = build_sampler(dist=dist,
                        shuffle=shuffle,
                        dataset=dataset,
                        num_gpus=num_gpus,
                        samples_per_gpu=samples_per_gpu,
                        workers_per_gpu=workers_per_gpu,
                        seed=seed, )
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    if isinstance(dataset, QueryAwareDataset):
        from .utils import query_support_collate_fn
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                query_support_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
    elif isinstance(dataset, NwayKshotDataset):
        from .dataloader_wrappers import NwayKshotDataloader
        from .utils import query_support_collate_fn

        # init query dataloader
        query_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                query_support_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # creat support dataset from query dataset and
        # sample batch index with same length as query dataloader
        support_dataset = copy.deepcopy(dataset)
        support_dataset.convert_query_to_support(
            len(query_data_loader) * num_gpus)

        (support_sampler, _, _) \
            = build_sampler(dist=dist,
                            shuffle=False,
                            dataset=support_dataset,
                            num_gpus=num_gpus,
                            samples_per_gpu=1,
                            workers_per_gpu=workers_per_gpu,
                            seed=seed,
                            )

        data_loader = NwayKshotDataloader(
            query_data_loader=query_data_loader,
            support_dataset=support_dataset,
            support_sampler=support_sampler,
            num_workers=num_workers,
            support_collate_fn=partial(
                query_support_collate_fn, samples_per_gpu=1),
            pin_memory=False,
            worker_init_fn=init_fn,
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


def build_sampler(dist, shuffle, dataset, num_gpus, samples_per_gpu,
                  workers_per_gpu, seed):
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
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    return sampler, batch_size, num_workers
