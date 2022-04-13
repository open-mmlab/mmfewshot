# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Union

import torch
from mmcls.core import DistOptimizerHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, build_optimizer, build_runner)
from mmcv.utils import ConfigDict, build_from_cfg
from torch.utils.data import Dataset

from mmfewshot.classification.core.evaluation import (DistMetaTestEvalHook,
                                                      MetaTestEvalHook)
from mmfewshot.classification.datasets.builder import (
    build_dataloader, build_dataset, build_meta_test_dataloader)
from mmfewshot.utils import get_root_logger


def train_model(model: Union[MMDataParallel, MMDistributedDataParallel],
                dataset: Dataset,
                cfg: ConfigDict,
                distributed: bool = False,
                validate: bool = False,
                timestamp: str = None,
                device: str = None,
                meta: Dict = None) -> None:
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        pin_memory=cfg.get('pin_memory', False),
        use_infinite_sampler=cfg.use_infinite_sampler)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cpu':
            warnings.warn(
                'The argument `device` is deprecated. To use cpu to train, '
                'please refers to https://mmclassification.readthedocs.io/en'
                '/latest/getting_started.html#train-a-model')
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    # Infinite sampler will return a infinite stream of index. It can NOT
    # be used in `EpochBasedRunner`, because the `EpochBasedRunner` will
    # enumerate the dataloader forever. Thus, `InfiniteEpochBasedRunner`
    # is designed to handle dataloader with infinite sampler.
    if cfg.use_infinite_sampler and cfg.runner['type'] == 'EpochBasedRunner':
        cfg.runner['type'] = 'InfiniteEpochBasedRunner'
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # build dataset and dataloader
        val_dataset = build_dataset(cfg.data.val)
        meta_test_cfg = cfg.data.val['meta_test_cfg']
        (support_data_loader, query_data_loader,
         all_data_loader) = build_meta_test_dataloader(val_dataset,
                                                       meta_test_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['meta_test_cfg'] = meta_test_cfg
        # register meta test hooks
        eval_hook = DistMetaTestEvalHook if distributed else MetaTestEvalHook
        runner.register_hook(
            eval_hook(
                support_data_loader,
                query_data_loader,
                all_data_loader,
                num_test_tasks=meta_test_cfg['num_episodes'],
                **eval_cfg),
            priority='LOW')

        # user-defined hooks
        if cfg.get('custom_hooks', None):
            custom_hooks = cfg.custom_hooks
            assert isinstance(
                custom_hooks, list
            ), f'custom_hooks expect list type, but got {type(custom_hooks)}'
            for hook_cfg in cfg.custom_hooks:
                assert isinstance(
                    hook_cfg, dict
                ), f'Each item in custom_hooks expects dict type, but ' \
                   f'got {type(hook_cfg)}'
                hook_cfg = hook_cfg.copy()
                priority = hook_cfg.pop('priority', 'NORMAL')
                hook = build_from_cfg(hook_cfg, HOOKS)
                runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
