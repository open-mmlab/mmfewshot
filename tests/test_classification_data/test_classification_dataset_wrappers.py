# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import numpy as np

from mmfewshot.classification.datasets import (EpisodicDataset,
                                               FewShotBaseDataset,
                                               MetaTestDataset)


@patch.multiple(FewShotBaseDataset, __abstractmethods__=set())
def construct_toy_dataset():
    FewShotBaseDataset.CLASSES = ('a', 'b', 'c', 'd', 'e', 'f', 'g')
    cat_ids_list = [i for i in range(7)] * 20
    data_infos = [dict(gt_label=np.array(i)) for i in cat_ids_list]
    FewShotBaseDataset.load_annotations = MagicMock(return_value=data_infos)
    dataset = FewShotBaseDataset(data_prefix='', pipeline=[])
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    return dataset, cat_ids_list


def test_episodic_dataset():
    toy_dataset, cat_ids_list = construct_toy_dataset()

    episodic_dataset_a = EpisodicDataset(
        toy_dataset,
        num_episodes=10,
        num_ways=5,
        num_shots=2,
        num_queries=3,
        episodes_seed=0)
    episodic_dataset_b = EpisodicDataset(
        toy_dataset,
        num_episodes=10,
        num_ways=5,
        num_shots=2,
        num_queries=3,
        episodes_seed=1)

    assert len(episodic_dataset_a) == 10 and len(episodic_dataset_a) == 10
    assert len(episodic_dataset_a[5]['support_data']) == 5 * 2
    assert len(episodic_dataset_a[5]['query_data']) == 5 * 3

    assert episodic_dataset_a[5]['query_data'] != episodic_dataset_b[5][
        'query_data']


def test_meta_test_dataset():
    toy_dataset, cat_ids_list = construct_toy_dataset()

    meta_dataset = MetaTestDataset(
        dataset=toy_dataset,
        num_episodes=10,
        num_ways=5,
        num_shots=2,
        num_queries=3,
        episodes_seed=0)
    test_set = meta_dataset.test_set()
    assert test_set._mode == 'test_set'
    assert len(test_set) == 140
    test_set.set_task_id(9)
    task9_class_id = test_set.get_task_class_ids()
    assert len(task9_class_id) == 5
    test_set.set_task_id(5)
    assert test_set.get_task_class_ids() != task9_class_id

    support_set = meta_dataset.support()
    assert support_set._mode == 'support'
    assert len(support_set) == 10

    query_set = meta_dataset.query()
    assert query_set._mode == 'query'
    assert len(query_set) == 15
