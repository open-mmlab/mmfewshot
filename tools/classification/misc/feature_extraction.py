"""Extracting features from pretrained backbone.

Example:
    # extract image from mini imagenet train set
    python ./tools/classification/feature_extraction.py \
        path/to/config path/to/checkpoint \
        mini_imagenet train --img_size 84
    # extract image from cub train+val+test set with image size of 224
    python ./tools/classification/feature_extraction.py \
        path/to/config path/to/checkpoint \
        cub train val test --img_size 224
"""

import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcls.apis.test import collect_results_cpu
from mmcls.models import build_classifier
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmfewshot.classification.datasets.builder import (build_dataloader,
                                                       build_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description='extract features from backbone.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument(
        'subset', type=str, nargs='+', help='subset of dataset')
    parser.add_argument(
        '--img-size', type=int, default=84, help='size of images')
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        default=32,
        help='number of samples in single gpu')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=0,
        help='number of workers in single gpu')
    parser.add_argument('--work-dir', help='the dir to save result')
    parser.add_argument(
        '--out-name',
        type=str,
        default=None,
        help='the file name to save result')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_dataset_cfg(dataset, subset, img_size):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    pipeline = [
        dict(type='LoadImageFromFile')
        if dataset != 'tiered_imagenet' else dict(type='LoadImageFromByte'),
        dict(type='Resize', size=(int(img_size * 1.15), -1)),
        dict(type='CenterCrop', crop_size=84),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', 'gt_label'])
    ]
    DATASETS = dict(
        cub=dict(
            type='CUBDataset',
            subset=subset,
            data_prefix='data/CUB_200_2011',
            pipeline=pipeline),
        mini_imagenet=dict(
            type='MiniImageNetDataset',
            subset=subset,
            data_prefix='data/mini_imagenet',
            pipeline=pipeline),
        tiered_imagenet=dict(
            type='TieredImageNetDataset',
            subset=subset,
            data_prefix='data/tiered_imagenet',
            pipeline=pipeline))
    assert DATASETS.get(dataset, None), 'please add custom dataset'
    return DATASETS[dataset]


def feature_extraction(model, dataloader, rank=0):
    feats_list, labels_list, img_metas_list = [], [], []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataloader))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            feats = model(img=data['img'], mode='extract_feat')
            feats_list.append(feats)
            labels_list.append(data['gt_label'])
            img_metas_list.extend(data['img_metas'].data[0])
            if rank == 0:
                prog_bar.update()
        feats = torch.cat(feats_list, dim=0).cpu()
        labels = torch.cat(labels_list, dim=0).cpu()
    return feats, labels, img_metas_list


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg.model.pretrained = None
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

        # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    data_cg = get_dataset_cfg(args.dataset, args.subset, args.img_size)
    dataset = build_dataset(data_cg)
    dataloader = build_dataloader(
        dataset,
        args.samples_per_gpu,
        args.workers_per_gpu,
        dist=distributed,
        round_up=True,
        shuffle=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    out_name = args.out_name
    if out_name is None:
        out_name = '-'.join([args.dataset] + args.subset +
                            [str(args.img_size), 'feats.pth'])
    out_path = osp.join(cfg.work_dir, out_name)
    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        feats, labels, img_metas_list = feature_extraction(model, dataloader)
        torch.save(
            {
                'feats': feats,
                'labels': labels,
                'img_metas_list': img_metas_list
            }, out_path)
        print(f'extracted features is saved to {out_path}')
    else:
        rank, world_size = get_dist_info()
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        feats, labels, img_metas_list = feature_extraction(
            model, dataloader, rank)
        feats_list = collect_results_cpu(feats, len(dataset))
        labels_list = collect_results_cpu(labels, len(dataset))
        img_metas_list = collect_results_cpu(img_metas_list, len(dataset))
        if rank == 0:
            torch.save(
                {
                    'feats': torch.stack(feats_list, dim=0),
                    'labels': torch.tensor(labels_list),
                    'img_metas_list': img_metas_list
                }, out_path)
            print(f'extracted features is saved to {out_path}')


if __name__ == '__main__':
    main()
