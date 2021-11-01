# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations
import os.path as osp
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from mmfewshot.utils import local_numpy_seed
from .builder import DATASETS


@DATASETS.register_module()
class EpisodicDataset:
    """A wrapper of episodic dataset.

    It will generate a list of support and query images indexes for each
    episode (support + query images). Every call of `__getitem__` will fetch
    and return (`num_ways` * `num_shots`) support images and (`num_ways` *
    `num_queries`) query images according to the generated images indexes.
    Note that all the episode indexes are generated at once using a specific
    random seed to ensure the reproducibility for same dataset.

    Args:
        dataset (:obj:`Dataset`): The dataset to be wrapped.
        num_episodes (int): Number of episodes. Noted that all episodes are
            generated at once and will not be changed afterwards. Make sure
            setting the `num_episodes` larger than your needs.
        num_ways (int): Number of ways for each episode.
        num_shots (int): Number of support data of each way for each episode.
        num_queries (int): Number of query data of each way for each episode.
        episodes_seed (int | None): A random seed to reproduce episodic
            indexes. If seed is None, it will use runtime random seed.
            Default: None.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_episodes: int,
                 num_ways: int,
                 num_shots: int,
                 num_queries: int,
                 episodes_seed: Optional[int] = None) -> None:
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self._len = len(self.dataset)
        self.CLASSES = dataset.CLASSES
        # using same episodes seed can generate same episodes for same dataset
        # it is designed for the reproducibility of meta train or meta test
        self.episodes_seed = episodes_seed
        self.episode_idxes, self.episode_class_ids = \
            self.generate_episodic_idxes()

    def generate_episodic_idxes(self) -> Tuple[List[Mapping], List[List[int]]]:
        episode_idxes, episode_class_ids = [], []
        class_ids = [i for i in range(len(self.CLASSES))]
        # using same episodes seed can generate same episodes for same dataset
        # it is designed for the reproducibility of meta train or meta test
        with local_numpy_seed(self.episodes_seed):
            for _ in range(self.num_episodes):
                np.random.shuffle(class_ids)
                sampled_cls = class_ids[:self.num_ways]
                episode_class_ids.append(sampled_cls)
                episodic_support_idx = []
                episodic_query_idx = []
                for i in range(self.num_ways):
                    shots = self.dataset.sample_shots_by_class_id(
                        sampled_cls[i], self.num_shots + self.num_queries)
                    episodic_support_idx += shots[:self.num_shots]
                    episodic_query_idx += shots[self.num_shots:]
                episode_idxes.append({
                    'support': episodic_support_idx,
                    'query': episodic_query_idx
                })
        return episode_idxes, episode_class_ids

    def __getitem__(self, idx: int) -> Dict:
        """Return a episode data at the same time.

        For `EpisodicDataset`, this function would return num_ways *
        num_shots support images and num_ways * num_queries query image.
        """

        return {
            'support_data':
            [self.dataset[i] for i in self.episode_idxes[idx]['support']],
            'query_data':
            [self.dataset[i] for i in self.episode_idxes[idx]['query']]
        }

    def __len__(self) -> int:
        return self.num_episodes

    def evaluate(self, *args, **kwargs) -> List:
        return self.dataset.evaluate(*args, **kwargs)

    def get_episode_class_ids(self, idx: int) -> List[int]:
        return self.episode_class_ids[idx]


@DATASETS.register_module()
class MetaTestDataset(EpisodicDataset):
    """A wrapper of the episodic dataset for meta testing.

    During meta test, the `MetaTestDataset` will be copied and converted into
    three mode: `test_set`, `support`, and `test`. Each mode of dataset will
    be used in different dataloader, but they share the same episode and image
    information.

    - In `test_set` mode, the dataset will fetch all images from the
      whole test set to extract features from the fixed backbone, which
      can accelerate meta testing.
    - In `support` or `query` mode, the dataset will fetch images
      according to the `episode_idxes` with the same `task_id`. Therefore,
      the support and query dataset must be set to the same `task_id` in
      each test task.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mode = 'test_set'
        self._task_id = 0
        self._with_cache_feats = False

    def with_cache_feats(self) -> bool:
        return self._with_cache_feats

    def set_task_id(self, task_id: int) -> None:
        """Query and support dataset use same task id to make sure fetch data
        from same episode."""
        self._task_id = task_id

    def __getitem__(self, idx: int) -> Dict:
        """Return data according to mode.

        For mode `test_set`, this function would return single image as regular
        dataset. For mode `support`, this function would return single support
        image of current episode. For mode `query`, this function would return
        single query image of current episode. If the dataset have cached the
        extracted features from fixed backbone, then the features will be
        return instead of image.
        """

        if self._mode == 'test_set':
            idx = idx
        elif self._mode == 'support':
            idx = self.episode_idxes[self._task_id]['support'][idx]
        elif self._mode == 'query':
            idx = self.episode_idxes[self._task_id]['query'][idx]

        if self._with_cache_feats:
            return {
                'feats': self.dataset.data_infos[idx]['feats'],
                'gt_label': self.dataset.data_infos[idx]['gt_label']
            }
        else:
            return self.dataset[idx]

    def get_task_class_ids(self) -> List[int]:
        return self.get_episode_class_ids(self._task_id)

    def test_set(self) -> MetaTestDataset:
        self._mode = 'test_set'
        return self

    def support(self) -> MetaTestDataset:
        self._mode = 'support'
        return self

    def query(self) -> MetaTestDataset:
        self._mode = 'query'
        return self

    def __len__(self) -> int:
        if self._mode == 'test_set':
            return len(self.dataset)
        elif self._mode == 'support':
            return self.num_ways * self.num_shots
        elif self._mode == 'query':
            return self.num_ways * self.num_queries

    def cache_feats(self, feats: Tensor, img_metas: Dict) -> None:
        """Cache extracted feats into dataset."""
        idx_map = {
            osp.join(data_info['img_prefix'],
                     data_info['img_info']['filename']): idx
            for idx, data_info in enumerate(self.dataset.data_infos)
        }
        # use filename as unique id
        for feat, img_meta in zip(feats, img_metas):
            idx = idx_map[img_meta['filename']]
            self.dataset.data_infos[idx]['feats'] = feat
        self._with_cache_feats = True
