# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mmfewshot.utils.infinite_sampler import (DistributedInfiniteGroupSampler,
                                              DistributedInfiniteSampler,
                                              InfiniteGroupSampler,
                                              InfiniteSampler)


class ExampleDataset(Dataset):

    def __init__(self):
        self.flag = np.array([0, 1], dtype=np.uint8)

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([idx]), img_metas=dict(idx=idx))
        return results

    def __len__(self):
        return 2


class ExampleDataset2(Dataset):

    def __init__(self):
        self.flag = np.array([0, 1, 1, 1], dtype=np.uint8)

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([idx]), img_metas=dict(idx=idx))
        return results

    def __len__(self):
        return 4


def test_infinite_sampler():
    dataset = ExampleDataset()
    sampler = InfiniteSampler(dataset=dataset, shuffle=False)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=1)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert 'img' in data
        assert 'img_metas' in data


def test_infinite_group_sampler():
    dataset = ExampleDataset()
    sampler = InfiniteGroupSampler(
        dataset=dataset, shuffle=False, samples_per_gpu=2)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=2)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert torch.allclose(data['img_metas']['idx'][0],
                              data['img_metas']['idx'][1])


def test_dist_infinite_sampler():
    dataset = ExampleDataset()
    sampler = DistributedInfiniteSampler(
        dataset=dataset, shuffle=False, num_replicas=2, rank=0)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=1)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert data['img'].item() == 0


def test_dist_group_infinite_sampler():
    dataset = ExampleDataset2()
    sampler = DistributedInfiniteGroupSampler(
        dataset=dataset,
        shuffle=False,
        num_replicas=2,
        rank=0,
        samples_per_gpu=2)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=2)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        if i % 2 == 0:
            assert torch.allclose(data['img_metas']['idx'],
                                  torch.tensor([0, 0]))
        else:
            assert torch.allclose(data['img_metas']['idx'],
                                  torch.tensor([2, 2]))
    sampler = DistributedInfiniteGroupSampler(
        dataset=dataset,
        shuffle=False,
        num_replicas=2,
        rank=1,
        samples_per_gpu=2)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=2)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert torch.allclose(data['img_metas']['idx'], torch.tensor([1, 3]))
