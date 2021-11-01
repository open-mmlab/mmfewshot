# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from typing import Dict, Optional

from mmcv.runner import Hook, Runner, get_dist_info
from torch.utils.data import DataLoader

from mmfewshot.classification.apis import (Z_SCORE, multi_gpu_meta_test,
                                           single_gpu_meta_test)


class MetaTestEvalHook(Hook):
    """Evaluation hook.

    Args:
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of all
            test data.
        num_test_tasks (int): Number of tasks for meta testing.
        interval (int): Evaluation interval (by epochs or iteration).
            Default: 1.
        by_epoch (bool): Epoch based runner or not. Default: True.
        meta_test_cfg (dict): Config for meta testing.
        confidence_interval (float): Confidence interval. Default: 0.95.
        save_best (bool): Whether to save best validated model.
            Default: True.
        key_indicator (str): The validation metric for selecting the
            best model. Default: 'accuracy_mean'.
        eval_kwargs : Any keyword argument to be used for evaluation.
    """

    def __init__(self,
                 support_dataloader: DataLoader,
                 query_dataloader: DataLoader,
                 test_set_dataloader: DataLoader,
                 num_test_tasks: int,
                 interval: int = 1,
                 by_epoch: bool = True,
                 meta_test_cfg: Optional[Dict] = None,
                 confidence_interval: float = 0.95,
                 save_best: bool = True,
                 key_indicator: str = 'accuracy_mean',
                 **eval_kwargs) -> None:
        if test_set_dataloader is None:
            dataloaders = [support_dataloader, query_dataloader]
        else:
            dataloaders = [
                support_dataloader, query_dataloader, test_set_dataloader
            ]
        for dataloader in dataloaders:
            if not isinstance(dataloader, DataLoader):
                raise TypeError(
                    'dataloader must be a pytorch DataLoader, but got'
                    f' {type(dataloader)}')
            assert dataloader.dataset.num_episodes == num_test_tasks, \
                'num_episodes of dataloader is mismatch with num_test_tasks '

        self.support_dataloader = support_dataloader
        self.query_dataloader = query_dataloader
        self.test_set_dataloader = test_set_dataloader

        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch
        self.num_test_tasks = num_test_tasks
        self.meta_test_cfg = meta_test_cfg
        assert confidence_interval in Z_SCORE.keys()
        self.confidence_interval = confidence_interval
        self.save_best = save_best
        self.best_score = 0.0
        self.key_indicator = key_indicator

    def before_run(self, runner: Runner) -> None:
        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating an empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())
            self.best_ckpt_path = runner.meta['hook_msgs'].get(
                'best_ckpt', None)

    def after_train_epoch(self, runner: Runner) -> None:
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        self.evaluate(runner)

    def after_train_iter(self, runner: Runner) -> None:
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        self.evaluate(runner)

    def evaluate(self, runner: Runner) -> Dict:
        meta_eval_results = single_gpu_meta_test(
            runner.model,
            self.num_test_tasks,
            self.support_dataloader,
            self.query_dataloader,
            self.test_set_dataloader,
            meta_test_cfg=self.meta_test_cfg,
            eval_kwargs=self.eval_kwargs,
            logger=runner.logger,
            confidence_interval=self.confidence_interval)
        if self.save_best:
            self._save_ckpt(runner, meta_eval_results[self.key_indicator])
        for name, val in meta_eval_results.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        return meta_eval_results

    def _save_ckpt(self, runner: Runner, key_score: float) -> None:
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        if self.best_score < key_score:
            self.best_score = key_score
            runner.meta['hook_msgs']['best_score'] = self.best_score
            runner.meta['hook_msgs']['ckpt_time'] = current

            if self.best_ckpt_path and osp.isfile(self.best_ckpt_path):
                os.remove(self.best_ckpt_path)

            best_ckpt_name = f'best_{self.key_indicator}.pth'
            self.best_ckpt_path = osp.join(runner.work_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'Best {self.key_indicator} is {self.best_score:0.4f} '
                f'at {cur_time} {cur_type}.')


class DistMetaTestEvalHook(MetaTestEvalHook):
    """Distributed evaluation hook."""

    def evaluate(self, runner: Runner) -> Dict:
        meta_eval_results = multi_gpu_meta_test(
            runner.model,
            self.num_test_tasks,
            self.support_dataloader,
            self.query_dataloader,
            self.test_set_dataloader,
            meta_test_cfg=self.meta_test_cfg,
            eval_kwargs=self.eval_kwargs,
            logger=runner.logger,
            confidence_interval=self.confidence_interval)
        rank, _ = get_dist_info()
        if rank == 0:
            if self.save_best:
                self._save_ckpt(runner, meta_eval_results[self.key_indicator])
            for name, val in meta_eval_results.items():
                runner.log_buffer.output[name] = val
            runner.log_buffer.ready = True
        return meta_eval_results
