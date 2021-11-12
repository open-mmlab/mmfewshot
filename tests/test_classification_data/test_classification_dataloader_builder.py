# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import numpy as np
from mmcv.utils import ConfigDict

from mmfewshot.classification.datasets import (CUBDataset, MetaTestDataset,
                                               build_dataloader,
                                               build_meta_test_dataloader)


@patch.multiple(CUBDataset, __abstractmethods__=set())
def construct_toy_dataset():
    CUBDataset.CLASSES = ('a', 'b', 'c', 'd', 'e', 'f', 'g')
    cat_ids_list = [i for i in range(7)] * 20
    data_infos = [dict(gt_label=np.array(i)) for i in cat_ids_list]
    CUBDataset.load_annotations = MagicMock(return_value=data_infos)
    dataset = CUBDataset(data_prefix='', pipeline=[])
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    dataset.sample_shots_by_class_id = MagicMock(
        side_effect=lambda class_id, idx: list(range(idx)))
    return dataset, cat_ids_list


def test_dataloader_builder():
    toy_dataset, cat_ids_list = construct_toy_dataset()

    dataloader = build_dataloader(
        toy_dataset,
        samples_per_gpu=2,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=True,
        round_up=True,
        use_infinite_sampler=False)
    assert len(dataloader) == 70
    # test infinite_sampler
    dataloader = build_dataloader(
        toy_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=True,
        round_up=True,
        use_infinite_sampler=True)
    assert len(dataloader) == 140


def test_meta_test_dataloader_builder():
    toy_dataset, cat_ids_list = construct_toy_dataset()
    num_ways, num_shots, num_queries = 5, 2, 3
    meta_dataset = MetaTestDataset(
        dataset=toy_dataset,
        num_episodes=10,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        episodes_seed=0)
    meta_test_cfg = ConfigDict(
        num_episodes=2000,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        # seed for generating meta test episodes
        episodes_seed=0,
        # whether to cache features in fixed-backbone methods for
        # testing acceleration.
        fast_test=True,
        test_set=dict(batch_size=16, num_workers=2),
        # worker initialization is a time consuming operation
        support=dict(batch_size=num_ways * num_shots, num_workers=0),
        query=dict(batch_size=num_ways * num_queries, num_workers=0))
    dataloaders = build_meta_test_dataloader(
        meta_dataset, meta_test_cfg=meta_test_cfg)
    assert len(dataloaders) == 3
    assert len(dataloaders[0].dataset) == 10
    assert len(dataloaders[1].dataset) == 15
    assert len(dataloaders[2].dataset) == 140
    assert dataloaders[0].dataset._task_id == 0
    dataloaders[0].dataset.set_task_id(1)
    assert dataloaders[0].dataset._task_id == 1
    # disable fast test
    meta_test_cfg = ConfigDict(
        num_episodes=2000,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        # seed for generating meta test episodes
        episodes_seed=0,
        # whether to cache features in fixed-backbone methods for
        # testing acceleration.
        fast_test=False,
        test_set=dict(batch_size=16, num_workers=2),
        # worker initialization is a time consuming operation
        support=dict(batch_size=num_ways * num_shots, num_workers=0),
        query=dict(batch_size=num_ways * num_queries, num_workers=0))
    dataloaders = build_meta_test_dataloader(
        meta_dataset, meta_test_cfg=meta_test_cfg)
    assert dataloaders[2] is None
