from typing import Dict, List

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Collect, DefaultFormatBundle


@PIPELINES.register_module()
class MultiScaleFormatBundle(DefaultFormatBundle):

    def __call__(self, results_list: List[Dict]) -> List[Dict]:
        """Transform and format common fields of each results in
        `results_list`.

        Args:
            results_list (list[dict]): List of result dict contains the data
                to convert.

        Returns:
            list[dict]: List of result dict contains the data that is formatted
                with default bundle.
        """
        for results in results_list:
            super().__call__(results)
        return results_list


@PIPELINES.register_module()
class MultiScaleCollect(Collect):

    def __call__(self, results_list: List[Dict]) -> Dict:
        """Collect all keys of each results in `results_list`.

        The keys in `meta_keys` will be converted to :obj:mmcv.DataContainer.
        A scale suffix also will be added to each key to specific from which
        scale of results.

        Args:
            results_list (list[dict]): List of result dict contains the data
                to collect.

        Returns:
            dict: The result dict contains the following keys

                - `{key}_scale_{i}` for i in 'num_scales' for key in`self.keys`
                - `img_metas_scale_{i}` for i in 'num_scales'
        """
        data = {}
        for i, results in enumerate(results_list):
            img_meta = {key: results[key] for key in self.meta_keys}
            data[f'img_metas_scale{i}'] = DC(img_meta, cpu_only=True)
            for key in self.keys:
                data[f'{key}_scale_{i}'] = results[key]
        return data
