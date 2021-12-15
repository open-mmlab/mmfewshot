# Copyright (c) OpenMMLab. All rights reserved.
import mmcls  # noqa: F401, F403
import mmcv
import mmdet  # noqa: F401, F403

from .classification import *  # noqa: F401, F403
from .detection import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .version import __version__, short_version


def digit_version(version_str):
    digit_version_ = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version_.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version_.append(int(patch_version[0]) - 1)
            digit_version_.append(int(patch_version[1]))
    return digit_version_


mmcv_minimum_version = '1.3.12'
mmcv_maximum_version = '1.5.0'
mmcv_version = digit_version(mmcv.__version__)


assert (digit_version(mmcv_minimum_version) <= mmcv_version
        <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

mmdet_minimum_version = '2.16.0'
mmdet_maximum_version = '2.20.0'
mmdet_version = digit_version(mmdet.__version__)


assert (digit_version(mmdet_minimum_version) <= mmdet_version
        <= digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version},\
     <={mmdet_maximum_version}.'

mmcls_minimum_version = '0.15.0'
mmcls_maximum_version = '0.19.0'
mmcls_version = digit_version(mmcls.__version__)


assert (digit_version(mmcls_minimum_version) <= mmcls_version
        <= digit_version(mmcls_maximum_version)), \
    f'MMCLS=={mmcls.__version__} is used but incompatible. ' \
    f'Please install mmcls>={mmcls_minimum_version},\
     <={mmcls_maximum_version}.'

__all__ = ['__version__', 'short_version']
