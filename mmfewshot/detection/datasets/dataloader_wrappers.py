from typing import Dict, Iterator

from torch.utils.data import DataLoader, Sampler

from .dataset_wrappers import NWayKShotDataset


class NWayKShotDataloader:
    """A dataloader wrapper.

    It Create a iterator to generate query and support
    batch simultaneously. Each batch contains query data
    and support data, and the lengths are batch_size and
    (num_support_ways * num_support_shots) respectively.

    Args:
        query_data_loader (DataLoader): DataLoader of query dataset
        support_dataset (:obj:`NWayKShotDataset`): Support datasets.
        support_sampler (Sampler): Sampler for support dataloader.
        num_workers (int): Num workers for support dataloader.
        support_collate_fn (callable): Collate function for support dataloader.
        pin_memory (bool): Pin memory for both support and query dataloader.
        worker_init_fn (callable): Worker init function for both
            support and query dataloader.
        shuffle_support_dataset (bool): Shuffle support dataset to generate
            new batch indexes. Default: False.
        kwargs: Any keyword argument to be used to initialize DataLoader.
    """

    def __init__(self,
                 query_data_loader: DataLoader,
                 support_dataset: NWayKShotDataset,
                 support_sampler: Sampler,
                 num_workers: int,
                 support_collate_fn: callable,
                 pin_memory: bool,
                 worker_init_fn: callable,
                 shuffle_support_dataset: bool = False,
                 **kwargs) -> None:
        self.dataset = query_data_loader.dataset
        self.query_data_loader = query_data_loader
        self.support_dataset = support_dataset
        self.support_sampler = support_sampler
        self.num_workers = num_workers
        self.support_collate_fn = support_collate_fn
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn
        self.shuffle_support_dataset = shuffle_support_dataset
        if self.shuffle_support_dataset:
            assert hasattr(
                self.support_dataset, 'shuffle_support'
            ), 'Support Dataset should support `shuffle_support`'
        self.kwargs = kwargs
        self.sampler = self.query_data_loader.sampler

        # support dataloader is initialized with batch_size 1 as default.
        # each batch contains (num_support_ways * num_support_shots) images,
        # since changing batch_size is equal to changing num_support_shots.
        self.support_data_loader = DataLoader(
            self.support_dataset,
            batch_size=1,
            sampler=self.support_sampler,
            num_workers=self.num_workers,
            collate_fn=self.support_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.kwargs)

    def __iter__(self) -> Iterator:
        if self.shuffle_support_dataset:
            # generate different support batch indexes for each epoch
            self.support_dataset.shuffle_support()
            # initialize support dataloader with batch_size 1
            # each batch contains (num_support_ways * num_support_shots)
            # images, the batch images are determined after generating
            # support batch indexes
            self.support_data_loader = DataLoader(
                self.support_dataset,
                batch_size=1,
                sampler=self.support_sampler,
                num_workers=self.num_workers,
                collate_fn=self.support_collate_fn,
                pin_memory=self.pin_memory,
                worker_init_fn=self.worker_init_fn,
                **self.kwargs)
        self.query_iter = iter(self.query_data_loader)
        self.support_iter = iter(self.support_data_loader)
        return self

    def __next__(self) -> Dict:
        # call query and support iterator
        query_data = self.query_iter.next()
        support_data = self.support_iter.next()
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self) -> int:
        return len(self.query_data_loader)


class TwoBranchDataloader:
    """A dataloader wrapper.

    It Create a iterator to iterate two different dataloader simultaneously.
    Note that `TwoBranchDataloader` dose not support epoch based training and
    the length of dataloader is decided by main dataset.

    Args:
        main_data_loader (DataLoader): DataLoader of main dataset.
        auxiliary_data_loader (DataLoader): DataLoader of auxiliary dataset.
    """

    def __init__(self, main_data_loader: DataLoader,
                 auxiliary_data_loader: DataLoader) -> None:
        self.dataset = main_data_loader.dataset
        self.main_data_loader = main_data_loader
        self.auxiliary_data_loader = auxiliary_data_loader

    def __iter__(self) -> Iterator:
        self.main_iter = iter(self.main_data_loader)
        self.auxiliary_iter = iter(self.auxiliary_data_loader)
        return self

    def __next__(self) -> Dict:
        # The iterator actually has infinite length, which can't
        # be used in epoch based training.
        try:
            main_data = next(self.main_iter)
        except StopIteration:
            self.main_iter = iter(self.main_data_loader)
            main_data = next(self.main_iter)
        try:
            auxiliary_data = next(self.auxiliary_iter)
        except StopIteration:
            self.auxiliary_iter = iter(self.auxiliary_data_loader)
            auxiliary_data = next(self.auxiliary_iter)
        return {'main_data': main_data, 'auxiliary_data': auxiliary_data}

    def __len__(self) -> int:
        return len(self.main_data_loader)
