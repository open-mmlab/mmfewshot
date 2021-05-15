from mmcls.datasets.builder import build_dataloader as build_cls_dataloader
from mmcls.datasets.builder import build_dataset as build_cls_dataset
from mmdet.datasets.builder import build_dataloader as build_det_dataloader
from mmdet.datasets.builder import build_dataset as build_det_dataset


def build_dataloader(*args, task_type='mmdet', **kwargs):

    if task_type == 'mmdet':
        data_loader = build_det_dataloader(*args, **kwargs)
    elif task_type == 'mmcls':
        data_loader = build_cls_dataloader(*args, **kwargs)
    return data_loader


def build_dataset(*args, task_type='mmdet', **kwargs):

    if task_type == 'mmdet':
        dataset = build_det_dataset(*args, **kwargs)
    elif task_type == 'mmcls':
        dataset = build_cls_dataset(*args, **kwargs)
    return dataset
