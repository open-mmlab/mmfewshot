# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock, patch

import mmcv.runner
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import obj_from_dict
from torch.utils.data import DataLoader, Dataset

from mmfewshot.classification.core import MetaTestEvalHook
from mmfewshot.classification.datasets import (BaseFewShotDataset,
                                               MetaTestDataset)


@patch.multiple(BaseFewShotDataset, __abstractmethods__=set())
def construct_toy_dataset():
    BaseFewShotDataset.CLASSES = ('a', 'b', 'c', 'd', 'e')
    cat_ids_list = [0, 1, 2, 3, 4] * 20
    data_infos = [
        dict(gt_label=np.array(i), img=torch.tensor(1), img_metas=[])
        for i in cat_ids_list
    ]
    BaseFewShotDataset.load_annotations = MagicMock(return_value=data_infos)
    dataset = BaseFewShotDataset(data_prefix='', pipeline=[])
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    return dataset, cat_ids_list


def toy_meta_test_dataset():
    toy_dataset, cat_ids_list = construct_toy_dataset()

    dataset = MetaTestDataset(
        dataset=toy_dataset,
        num_episodes=10,
        num_ways=1,
        num_shots=1,
        num_queries=1,
        episodes_seed=0)
    dataset.evaluate = MagicMock(return_value=dict(accuracy=0.9))
    return dataset


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.meta_test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)
        # device_indicator is used to record runtime device
        self.register_buffer('device_indicator', torch.empty(0))

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)

    def device(self) -> torch.device:
        return self.device_indicator.device

    def get_device(self):
        return self.device_indicator.get_device()

    def before_meta_test(self, meta_test_cfg, **kwargs):
        pass

    def before_forward_support(self, **kwargs):
        pass

    def before_forward_query(self, **kwargs):
        pass


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


def test_iter_eval_hook():
    test_set_loader = DataLoader(
        toy_meta_test_dataset(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    query_loader = DataLoader(
        toy_meta_test_dataset().query(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    support_loader = DataLoader(
        toy_meta_test_dataset().support(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    model = ExampleModel()
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    # test EvalHook
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = MetaTestEvalHook(
            support_loader,
            query_loader,
            test_set_loader,
            interval=1,
            by_epoch=False,
            num_test_tasks=10,
            meta_test_cfg=dict(support={}, query={}))
        runner = mmcv.runner.IterBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger(),
            max_iters=1)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)


def test_epoch_eval_hook():
    test_set_loader = DataLoader(
        toy_meta_test_dataset(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    query_loader = DataLoader(
        toy_meta_test_dataset().query(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    support_loader = DataLoader(
        toy_meta_test_dataset().support(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    model = ExampleModel()
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    # test EvalHook
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = MetaTestEvalHook(
            support_loader,
            query_loader,
            test_set_loader,
            num_test_tasks=10,
            meta_test_cfg=dict(support={}, query={}))
        runner = mmcv.runner.EpochBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger(),
            max_epochs=1)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)


def test_resume_eval_hook():
    test_set_loader = DataLoader(
        toy_meta_test_dataset(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    query_loader = DataLoader(
        toy_meta_test_dataset().query(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    support_loader = DataLoader(
        toy_meta_test_dataset().support(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)
    model = ExampleModel()
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    # test EvalHook
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = MetaTestEvalHook(
            support_loader,
            query_loader,
            test_set_loader,
            num_test_tasks=10,
            meta_test_cfg=dict(support={}, query={}))
        runner = mmcv.runner.EpochBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger(),
            max_epochs=1)
        runner.register_hook(eval_hook)
        runner.meta = {'best_score': 99.0}
        runner.run([loader], [('train', 1)], 1)
        assert eval_hook.best_score == 99.0
