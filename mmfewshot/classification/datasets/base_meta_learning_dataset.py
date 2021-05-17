# jsut an example
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.imagenet import ImageNet


@DATASETS.register_module()
class BaseMetaLearingDataset(ImageNet):
    pass
