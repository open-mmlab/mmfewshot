import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from mmdet.datasets.builder import DATASETS

from .few_shot_custom import FewShotCustomDataset


@DATASETS.register_module()
class FewShotVOCDataset(FewShotCustomDataset):
    """VOC dataset for few shot detection.

    FewShotVOCDataset allow annotation mask during loading annotation.
    The annotation can be loaded from image id or image path. For example:

    .. code-block:: none

            ann_image_id.txt:
                000001
                000002

            ann_image_path.txt:
                VOC2007/JPEGImages/000001.jpg
                VOC2007/JPEGImages/000002.jpg

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field. Default: None.
    """

    def __init__(self, min_size=None, **kwargs):
        assert self.CLASSES or kwargs.get(
            'classes', None), 'CLASSES in `XMLDataset` can not be None.'
        self.min_size = min_size
        super(FewShotVOCDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        img_names = mmcv.list_from_file(ann_file)
        for img_name in img_names:
            # ann file in image path format
            if 'VOC2007' in img_name:
                dataset_year = 'VOC2007'
                img_id = img_name.split('/')[-1].split('.')[0]
                filename = img_name
            # ann file in image path format
            elif 'VOC2012' in img_name:
                dataset_year = 'VOC2012'
                img_id = img_name.split('/')[-1].split('.')[0]
                filename = img_name
            # ann file in image id format
            elif 'VOC2007' in ann_file:
                dataset_year = 'VOC2007'
                img_id = img_name
                filename = f'VOC2007/JPEGImages/{img_name}.jpg'
            # ann file in image id format
            elif 'VOC2012' in ann_file:
                dataset_year = 'VOC2012'
                img_id = img_name
                filename = f'VOC2012/JPEGImages/{img_name}.jpg'
            else:
                raise ValueError('Cannot infer dataset year from img_prefix')

            xml_path = osp.join(self.img_prefix, dataset_year, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, dataset_year,
                                    'JPEGImages', '{}.jpg'.format(img_id))
                img = mmcv.imread(img_path)
                width, height = img.size
            # save annotation infos into data infos, because not all the
            # annotations will be used for training and the used annotations
            # should stay the same anytime during training.
            ann_info = self._get_ann_info(dataset_year, img_id)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))
        return data_infos

    def _get_ann_info(self, dataset_year, img_id):
        """Get annotation from XML file by img_id.

        Args:
            dataset_year (str): Year of voc dataset. Options are
                'VOC2007', 'VOC2012'
            img_id (str): Id of image.

        Returns:
            dict: Annotation info of specified id with specified class.
        """

        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        xml_path = osp.join(self.img_prefix, dataset_year, 'Annotations',
                            f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int).tolist()
                if len(cat_ids) > 0:
                    valid_inds.append(i)
            else:
                valid_inds.append(i)
        return valid_inds
