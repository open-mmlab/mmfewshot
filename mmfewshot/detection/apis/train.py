# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import ConfigDict, build_from_cfg
from mmdet.core import DistEvalHook, EvalHook

from mmfewshot.detection.core import (QuerySupportDistEvalHook,
                                      QuerySupportEvalHook)
from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.utils import get_root_logger


def train_detector(model: nn.Module,
                   dataset: Iterable,
                   cfg: ConfigDict,
                   distributed: bool = False,
                   validate: bool = False,
                   timestamp: Optional[str] = None,
                   meta: Optional[Dict] = None) -> None:
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            data_cfg=copy.deepcopy(cfg.data),
            use_infinite_sampler=cfg.use_infinite_sampler) for ds in dataset
    ]

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
        # Please use MMCV >= 1.4.4 for CPU training!
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
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
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
        # currently only support single images testing
        samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        assert samples_per_gpu == 1, \
            'currently only support single images testing'

        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'

        # Prepare `model_init` dataset for model initialization. In most cases,
        # the `model_init` dataset contains support images and few shot
        # annotations. The meta-learning based detectors will extract the
        # features from images and save them as part of model parameters.
        # The `model_init` dataset can be mutually configured or
        # randomly selected during runtime.
        if cfg.data.get('model_init', None) is not None:
            # The randomly selected few shot support during runtime can not be
            # configured offline. In such case, the copy datasets are designed
            # to directly copy the randomly generated support set for model
            # initialization. The copy datasets copy the `data_infos` by
            # passing it as argument and other arguments can be different
            # from training dataset.
            if cfg.data.model_init.pop('copy_from_train_dataset', False):
                if cfg.data.model_init.ann_cfg is not None:
                    warnings.warn(
                        'model_init dataset will copy support '
                        'dataset used for training and original '
                        'ann_cfg will be discarded', UserWarning)
                # modify dataset type to support copying data_infos operation
                cfg.data.model_init.type = \
                    get_copy_dataset_type(cfg.data.model_init.type)
                if not hasattr(dataset[0], 'get_support_data_infos'):
                    raise NotImplementedError(
                        f'`get_support_data_infos` is not implemented '
                        f'in {dataset[0].__class__.__name__}.')
                cfg.data.model_init.ann_cfg = [
                    dict(data_infos=dataset[0].get_support_data_infos())
                ]
            # The `model_init` dataset will be saved into checkpoint, which
            # allows model to be initialized with these data as default, if
            # the config of data is not be overwritten during testing.
            cfg.checkpoint_config.meta['model_init_ann_cfg'] = \
                cfg.data.model_init.ann_cfg
            samples_per_gpu = cfg.data.model_init.pop('samples_per_gpu', 1)
            workers_per_gpu = cfg.data.model_init.pop('workers_per_gpu', 1)
            model_init_dataset = build_dataset(cfg.data.model_init)
            # Noted that `dist` should be FALSE to make all the models on
            # different gpus get same data results in same initialized models.
            model_init_dataloader = build_dataloader(
                model_init_dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=workers_per_gpu,
                dist=False,
                shuffle=False)

            # eval hook for meta-learning based query-support detector, it
            # supports model initialization before regular evaluation.
            eval_hook = QuerySupportDistEvalHook \
                if distributed else QuerySupportEvalHook
            runner.register_hook(
                eval_hook(model_init_dataloader, val_dataloader, **eval_cfg),
                priority='LOW')
        else:
            # for the fine-tuned based methods, the evaluation is the
            # same as mmdet.
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')

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
