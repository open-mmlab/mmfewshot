from mmcls.apis.test import multi_gpu_test as cls_multi_gpu_test
from mmcls.apis.test import single_gpu_test as cls_single_gpu_test
from mmdet.apis.test import multi_gpu_test as det_multi_gpu_test
from mmdet.apis.test import single_gpu_test as det_single_gpu_test


def single_gpu_test(*args, task_type='mmdet', **kwargs):
    if task_type == 'mmdet':
        return det_single_gpu_test(*args, **kwargs)
    elif task_type == 'mmcls':
        return cls_single_gpu_test(*args, **kwargs)
    else:
        raise NotImplementedError


def multi_gpu_test(*args, task_type='mmdet', **kwargs):
    if task_type == 'mmdet':
        return det_multi_gpu_test(*args, **kwargs)
    elif task_type == 'mmcls':
        return cls_multi_gpu_test(*args, **kwargs)
    raise NotImplementedError
