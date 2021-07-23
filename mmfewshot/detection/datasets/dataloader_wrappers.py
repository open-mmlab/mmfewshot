from torch.utils.data import DataLoader


class NwayKshotDataloader(object):
    """A dataloader wrapper of NwayKshotDataset dataset. Create a iterator to
    generate query and support batch simultaneously. Each batch return a batch.

    of query data (batch_size) and support data (support_way * support_shot).

    Args:
        query_data_loader (obj:`DataLoader`): DataLoader of query dataset
        support_dataset (list[:obj:`NwayKshotDataset`]): Support datasets.
        support_sampler (Sampler): Sampler for support dataloader.
        num_workers (int): Num workers for support dataloader.
        support_collate_fn (callable): Collate function for support dataloader.
        pin_memory (bool): Pin memory for both support and query dataloader.
        worker_init_fn (callable): Worker init function for both
            support and query dataloader.
        shuffle_support_dataset (bool): Shuffle support dataset to generate
            new batch indexes. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader.
    """

    def __init__(self,
                 query_data_loader,
                 support_dataset,
                 support_sampler,
                 num_workers,
                 support_collate_fn,
                 pin_memory,
                 worker_init_fn,
                 shuffle_support_dataset=False,
                 **kwargs):
        self.dataset = query_data_loader.dataset
        self.query_data_loader = query_data_loader
        self.support_dataset = support_dataset
        self.support_sampler = support_sampler
        self.num_workers = num_workers
        self.support_collate_fn = support_collate_fn
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn
        self.shuffle_support_dataset = shuffle_support_dataset
        self.kwargs = kwargs
        self.sampler = self.query_data_loader.sampler
        self.support_data_loader = DataLoader(
            self.support_dataset,
            batch_size=1,
            sampler=self.support_sampler,
            num_workers=self.num_workers,
            collate_fn=self.support_collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.kwargs)

    def __iter__(self):
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

    def __next__(self):
        # call query and support iterator
        query_data = self.query_iter.next()
        support_data = self.support_iter.next()
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self):
        return len(self.query_data_loader)
