# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from .base import BaseFewShotDataset

TRAIN_CLASSES = [
    'n02074367', 'n03047690', 'n03854065', 'n02089867', 'n02105505',
    'n01704323', 'n04604644', 'n03676483', 'n01558993', 'n07697537',
    'n04509417', 'n02101006', 'n02165456', 'n13133613', 'n02747177',
    'n02966193', 'n03924679', 'n04275548', 'n02113712', 'n03838899',
    'n02091831', 'n03220513', 'n07747607', 'n03998194', 'n02108089',
    'n09246464', 'n04251144', 'n02111277', 'n04435653', 'n03207743',
    'n04389033', 'n03337140', 'n03908618', 'n02606052', 'n01770081',
    'n01910747', 'n03062245', 'n02108551', 'n03017168', 'n04258138',
    'n03888605', 'n04596742', 'n07584110', 'n02687172', 'n03476684',
    'n04243546', 'n02795169', 'n02457408', 'n04515003', 'n06794110',
    'n01532829', 'n01843383', 'n13054560', 'n04067472', 'n03347037',
    'n04612504', 'n03400231', 'n01749939', 'n02823428', 'n04296562',
    'n03527444', 'n04443257', 'n02108915', 'n02120079'
]
VAL_CLASSES = [
    'n02138441', 'n02981792', 'n02174001', 'n03535780', 'n03770439',
    'n03773504', 'n02950826', 'n03980874', 'n02114548', 'n03584254',
    'n02091244', 'n03417042', 'n02971356', 'n01855672', 'n09256479',
    'n03075370'
]
TEST_CLASSES = [
    'n02110341', 'n01981276', 'n07613480', 'n02129165', 'n04418357',
    'n02443484', 'n03127925', 'n01930112', 'n03272010', 'n03146219',
    'n04146614', 'n03775546', 'n04522168', 'n02099601', 'n02871525',
    'n02110063', 'n02219486', 'n02116738', 'n04149813', 'n03544143'
]


@DATASETS.register_module()
class MiniImageNetDataset(BaseFewShotDataset):
    """MiniImageNet dataset for few shot classification.

    Args:
        subset (str| list[str]): The classes of whole dataset are split into
            three disjoint subset: train, val and test. If subset is a string,
            only one subset data will be loaded. If subset is a list of
            string, then all data of subset in list will be loaded.
            Options: ['train', 'val', 'test']. Default: 'train'.
        file_format (str): The file format of the image. Default: 'JPEG'
    """

    resource = 'https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet'  # noqa

    TRAIN_CLASSES = TRAIN_CLASSES
    VAL_CLASSES = VAL_CLASSES
    TEST_CLASSES = TEST_CLASSES

    def __init__(self,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 file_format: str = 'JPEG',
                 *args,
                 **kwargs):
        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
        self.file_format = file_format
        super().__init__(*args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Three types of input
                will correspond to different processing logics:

                - If `classes` is a tuple or list, it will override the
                  CLASSES predefined in the dataset.
                - If `classes` is None, we directly use pre-defined CLASSES
                  will be used by the dataset.
                - If `classes` is a string, it is the path of a classes file
                  that contains the name of all classes. Each line of the file
                  contains a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += self.TRAIN_CLASSES
                elif subset_ == 'val':
                    class_names += self.VAL_CLASSES
                elif subset_ == 'test':
                    class_names += self.TEST_CLASSES
                else:
                    raise ValueError(f'invalid subset {subset_} only '
                                     f'support train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def load_annotations(self) -> List:
        """Load annotation according to the classes subset."""
        img_file_list = {
            class_name: sorted(
                os.listdir(osp.join(self.data_prefix, 'images', class_name)),
                key=lambda x: int(x.split('.')[0].split('_')[-1]))
            for class_name in self.CLASSES
        }
        data_infos = []
        for subset_ in self.subset:
            ann_file = osp.join(self.data_prefix, f'{subset_}.csv')
            assert osp.exists(ann_file), \
                f'Please download ann_file through {self.resource}.'
            with open(ann_file, 'r') as f:
                for i, line in enumerate(f):
                    # skip file head
                    if i == 0:
                        continue
                    filename, class_name = line.strip().split(',')
                    filename = img_file_list[class_name][
                        int(filename.split('.')[0][-5:]) - 1]
                    gt_label = self.class_to_idx[class_name]
                    info = {
                        'img_prefix':
                        osp.join(self.data_prefix, 'images', class_name),
                        'img_info': {
                            'filename': filename
                        },
                        'gt_label':
                        np.array(gt_label, dtype=np.int64)
                    }
                    data_infos.append(info)
        return data_infos
