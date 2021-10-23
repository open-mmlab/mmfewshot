import os.path as osp
from typing import Dict

import mmcv
import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import LoadImageFromFile


@PIPELINES.register_module()
class LoadImageFromBytes(LoadImageFromFile):
    """Load an image from bytes."""

    def __call__(self, results: Dict) -> Dict:
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        if results.get('img_bytes', None) is None:
            img_bytes = self.file_client.get(filename)
        else:
            img_bytes = results.pop('img_bytes')
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
