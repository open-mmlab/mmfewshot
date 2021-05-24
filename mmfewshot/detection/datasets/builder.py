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

from .dataset_wrappers import MergeDataset, NwayKshotDataset, QueryAwareDataset


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)
    merge_dataset = cfg.get('merge_dataset', False)
    ann_shot_filter = cfg.get('ann_shot_filter', None)

    if ann_shot_filter is not None:
        assert merge_dataset, 'using ann shot filter to load ann file ' \
              'in FewShotDataset, merge_dataset should be set to True.'

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        if 'merge_dataset' in data_cfg:
            data_cfg.pop('merge_dataset')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        if isinstance(ann_shot_filter, (list, tuple)):
            data_cfg['ann_shot_filter'] = ann_shot_filter[i]

        datasets.append(build_dataset(data_cfg, default_args))
    if merge_dataset:
        return MergeDataset(datasets)
    else:
        return ConcatDataset(datasets, separate_eval)


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
            build_dataset(cfg['dataset'], default_args), cfg['support_way'],
            cfg['support_shot'])
    elif cfg['type'] == 'NwayKshotDataset':
        dataset = NwayKshotDataset(
            build_dataset(cfg['dataset'], default_args), cfg['support_way'],
            cfg['support_shot'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
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
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int): Random seed.
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
        # pre sample batch index with same length as query dataloader
        support_dataset = copy.deepcopy(dataset)
        support_dataset.convert_query_to_support(len(query_data_loader))

        (support_sampler, _, _) \
            = build_sampler(dist=dist,
                            shuffle=shuffle,
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
