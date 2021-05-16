from mmcls.datasets.builder import build_dataloader as build_cls_dataloader
from mmcls.datasets.builder import build_dataset as build_cls_dataset
from mmdet.datasets.builder import build_dataloader as build_det_dataloader
from mmdet.datasets.builder import build_dataset as build_det_dataset


def build_dataloader(dataset=None, task_type='mmdet', round_up=True, **kwargs):
    # TODO: identify how to bulid the dataloader via the type of dataset
    # just an example
    # if isinstance(dataset,base_meta_learning_dataset):
    # data_loader=build_det_metalearning_dataloader(dataset=dataset, **kwargs)
    if task_type == 'mmdet':
        data_loader = build_det_dataloader(dataset=dataset, **kwargs)
    elif task_type == 'mmcls':
        data_loader = build_cls_dataloader(
            dataset=dataset, round_up=round_up, **kwargs)
    else:
        raise NotImplementedError
    return data_loader


def build_dataset(*args, task_type='mmdet', **kwargs):

    if task_type == 'mmdet':
        dataset = build_det_dataset(*args, **kwargs)
    elif task_type == 'mmcls':
        dataset = build_cls_dataset(*args, **kwargs)
    else:
        raise NotImplementedError
    return dataset


# TODO: check whether det and cls can use same dataloader for meta_learnig
def build_det_metalearning_dataloader():
    pass
