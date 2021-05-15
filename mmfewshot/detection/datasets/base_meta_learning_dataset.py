# jsut an example
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class BaseMetaLearingDataset(CustomDataset):
    pass
