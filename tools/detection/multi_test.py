"""Test multiple run few shot models."""
import argparse
import copy
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmfewshot  # noqa: F401, F403
from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.detection.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) models for multi train')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'times', type=int, help='number of experiments to test')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='latest',
        help='name of checkpoint to test')
    parser.add_argument(
        '--start',
        default=0,
        type=int,
        help='number of resume experiment times')
    parser.add_argument(
        '--work-dir', default=None, help='work directory for experiments')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset '
        'of specific task_type, e.g., "bbox","segm", "proposal" for '
        'COCO, and "mAP", "recall" for PASCAL VOC in'
        'MMDet or "accuracy", "precision", "recall", "f1_score", '
        '"support" for single label dataset, and "mAP", "CP", "CR",'
        '"CF1", "OP", "OR", "OF1" for '
        'multi-label dataset in MMCLS')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
        args.cfg_options = args.options
    return args


def main():
    args = parse_args()

    if args.work_dir is None:
        args.work_dir = os.path.join(
            './work_dirs',
            os.path.splitext(os.path.basename(args.config))[0])

    assert args.out or args.eval, (
        'Please specify at least one operation (save/eval the '
        'results / save the results) with the argument "--out", "--eval"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    base_cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        base_cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if base_cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**base_cfg['custom_imports'])
    # set cudnn_benchmark
    if base_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    base_cfg.model.pretrained = None

    # currently only support single images testing
    samples_per_gpu = base_cfg.data.test.pop('samples_per_gpu', 1)
    assert samples_per_gpu == 1, 'currently only support single images testing'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **base_cfg.dist_params)
    rank, _ = get_dist_info()
    eval_result_list = []
    # build the dataloader
    dataset = build_dataset(base_cfg.data.test)
    # currently only support single images testing
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=base_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    for i in range(args.start, args.times):
        cfg = copy.deepcopy(base_cfg)
        work_dir = os.path.join(args.work_dir, f'times_{i}')
        cfg.work_dir = work_dir
        # pop frozen_parameters
        cfg.model.pop('frozen_parameters', None)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model)
        checkpoint_path = os.path.join(work_dir, f'{args.checkpoint}.pth')
        checkpoint = load_checkpoint(
            model, checkpoint_path, map_location='cpu')

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        # for meta-learning methods which require support template dataset
        # for model initialization.
        if cfg.data.get('model_init', None) is not None:
            cfg.data.model_init.pop('copy_from_train_dataset')
            model_init_samples_per_gpu = cfg.data.model_init.pop(
                'samples_per_gpu', 1)
            model_init_workers_per_gpu = cfg.data.model_init.pop(
                'workers_per_gpu', 1)
            if cfg.data.model_init.get('ann_cfg', None) is None:
                assert checkpoint['meta'].get('model_init_ann_cfg',
                                              None) is not None
                cfg.data.model_init.type = \
                    get_copy_dataset_type(cfg.data.model_init.type)
                cfg.data.model_init.ann_cfg = \
                    checkpoint['meta']['model_init_ann_cfg']
            model_init_dataset = build_dataset(cfg.data.model_init)
            # disable dist to make all rank get same data
            model_init_dataloader = build_dataloader(
                model_init_dataset,
                samples_per_gpu=model_init_samples_per_gpu,
                workers_per_gpu=model_init_workers_per_gpu,
                dist=False,
                shuffle=False)

        # old versions did not save class info in checkpoints,
        # this walkaround is for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            show_kwargs = dict(show_score_thr=args.show_score_thr)
            if cfg.data.get('model_init', None) is not None:
                from mmfewshot.detection.apis import (single_gpu_model_init,
                                                      single_gpu_test)
                single_gpu_model_init(model, model_init_dataloader)
            else:
                from mmdet.apis.test import single_gpu_test
            outputs = single_gpu_test(model, data_loader, args.show,
                                      args.show_dir, **show_kwargs)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            if cfg.data.get('model_init', None) is not None:
                from mmfewshot.detection.apis import (multi_gpu_model_init,
                                                      multi_gpu_test)
                multi_gpu_model_init(model, model_init_dataloader)
            else:
                from mmdet.apis.test import multi_gpu_test
            outputs = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
            )
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect',
                        'save_best', 'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                eval_result = dataset.evaluate(outputs, **eval_kwargs)
                print(eval_result)
                eval_result_list.append(eval_result)

            if i == (args.times - 1):
                num_results = len(eval_result_list)
                if num_results == 0:
                    print('zero experiment tested')
                else:
                    avg_results = dict()
                    for k in eval_result_list[0].keys():
                        avg_results[f'Avg {k}'] = sum([
                            eval_result_list[i][k] for i in range(num_results)
                        ]) / num_results
                    mmcv.dump(avg_results,
                              os.path.join(args.work_dir, 'avg_result.json'))
                    print(avg_results)


if __name__ == '__main__':
    main()
