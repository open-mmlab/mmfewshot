import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_recalls
from mmdet.datasets.builder import DATASETS

from mmfewshot.detection.core import eval_map, voc_tpfp_fn
from .few_shot_custom import FewShotCustomDataset

# predefined classes split for few shot setting
VOC_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat',
                        'chair', 'diningtable', 'dog', 'horse', 'person',
                        'pottedplant', 'sheep', 'train', 'tvmonitor', 'bird',
                        'bus', 'cow', 'motorbike', 'sofa'),
    ALL_CLASSES_SPLIT2=('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                        'chair', 'diningtable', 'dog', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'train', 'tvmonitor',
                        'aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    ALL_CLASSES_SPLIT3=('aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car',
                        'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'person', 'pottedplant', 'train', 'tvmonitor', 'boat',
                        'cat', 'motorbike', 'sheep', 'sofa'),
    NOVEL_CLASSES_SPLIT1=('bird', 'bus', 'cow', 'motorbike', 'sofa'),
    NOVEL_CLASSES_SPLIT2=('aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    NOVEL_CLASSES_SPLIT3=('boat', 'cat', 'motorbike', 'sheep', 'sofa'),
    BASE_CLASSES_SPLIT1=('aeroplane', 'bicycle', 'boat', 'bottle', 'car',
                         'cat', 'chair', 'diningtable', 'dog', 'horse',
                         'person', 'pottedplant', 'sheep', 'train',
                         'tvmonitor'),
    BASE_CLASSES_SPLIT2=('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                         'chair', 'diningtable', 'dog', 'motorbike', 'person',
                         'pottedplant', 'sheep', 'train', 'tvmonitor'),
    BASE_CLASSES_SPLIT3=('aeroplane', 'bicycle', 'bird', 'bottle', 'bus',
                         'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                         'person', 'pottedplant', 'train', 'tvmonitor'))


@DATASETS.register_module()
class FewShotVOCDataset(FewShotCustomDataset):
    """VOC dataset for few shot detection.

    Args:
        classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load predefined classes in FewShotCocoDataset.
            For example: 'BASE_CLASSES'.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used for each base
            class. If is None, all annotation will be used. Default: None.
        ann_shot_filter (dict | None): If set None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
            If not None, annotation shot filter will specific which class and
            the maximum number of instances to load from annotation file.
            For example: {'dog': 10, 'person': 5}. Default: None.
        min_bbox_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_bbox_size``, it would be add to ignored field. Default: None.
        use_difficult (bool): Whether use the difficult annotation or not.
            Default: False.
        min_bbox_area_filter (float | None):  Filter images with bbox whose
            area smaller `min_bbox_area_filter`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
    """

    def __init__(self,
                 classes=None,
                 num_novel_shots=None,
                 num_base_shots=None,
                 ann_shot_filter=None,
                 min_bbox_size=None,
                 use_difficult=False,
                 min_bbox_area_filter=None,
                 dataset_name=None,
                 test_mode=False,
                 **kwargs):
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.SPLIT = VOC_SPLIT
        assert classes is not None, f'{self.dataset_name}: classes ' \
                                    f'in `FewShotVOCDataset` can not be None.'

        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(), \
                f'{self.dataset_name}: not a predefine classes' \
                f' or split in VOC_SPLIT'
            self.CLASSES = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not ' \
                    f'have novel instances'
            elif 'NOVEL_CLASSES' in classes:
                assert num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not ' \
                    f'have base instances'
            self.split = classes[-1]

        # configure ann_shot_filter by num_novel_shots and num_base_shots
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area_filter = min_bbox_area_filter
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is not None or num_base_shots is not None, \
                f'{self.dataset_name}: can not config ann_shot_filter ' \
                f'and num_novel_shots/num_base_shots at the same time.'

        self.min_bbox_size = min_bbox_size
        self.use_difficult = use_difficult
        super(FewShotVOCDataset, self).__init__(
            classes=self.CLASSES,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def _create_ann_shot_filter(self):
        """generate ann_shot_filter by novel and base classes."""
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT['NOVEL_CLASSES_SPLIT' + self.split]:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT['BASE_CLASSES_SPLIT' + self.split]:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg):
        """Load annotation from two type of ann_cfg.

           - type of 'ann_file': annotation txt (image id or image path)
                with or without specific classes.
           - type of 'saved_dataset': saved dataset json.

           Example:

           [dict(type='ann_file', ann_file='path/to/ann.txt'),

           dict(type='ann_file', ann_file='path/to/dog.txt',
                ann_classes=['dog', 'person']),

           dict(type='saved_dataset', ann_file='path/to/saved_data.json')]

        Args:
            ann_cfg (list[dict]): Config of annotations.

        Returns:
            list[dict]: Annotation information.
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                ann_classes = ann_cfg_.get('ann_classes', None)
                if ann_classes is not None:
                    for c in ann_classes:
                        assert c in self.CLASSES, \
                            f'{self.dataset_name}: ann_classes ' \
                            f'must in dataset classes.'
                else:
                    ann_classes = self.CLASSES
                data_infos += self.load_annotations_xml(
                    ann_cfg_['ann_file'], ann_classes)
            else:
                raise ValueError(
                    f'{self.dataset_name}: not support '
                    f'annotation type {ann_cfg_["type"]} in ann_cfg.')

        return data_infos

    def load_annotations_xml(self, ann_file, classes=None):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
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
            ann_info = self._get_xml_ann_info(dataset_year, img_id, classes)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))
        return data_infos

    def _get_xml_ann_info(self, dataset_year, img_id, classes=None):
        """Get annotation from XML file by img_id.

        Args:
            dataset_year (str): Year of voc dataset. Options are
                'VOC2007', 'VOC2012'
            img_id (str): Id of image.
            classes (list): Specific classes to load form xml file.
                If set to None, it will use classes of whole dataset.
                Default: None.
        Returns:
            dict: Annotation info of specified id with specified class.
        """
        if classes is None:
            classes = self.CLASSES
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
            if name not in classes:
                continue
            label = self.cat2label[name]
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')

            # It should be noted that in the original mmdet implementation,
            # the four coordinates are reduced by 1 when the annotation
            # is parsed. Here we following detectron2, only xmin and ymin
            # will be reduced by 1 during training. The groundtruth used for
            # evaluation or testing keep consisent with original xml
            # annotation file and the xmin and ymin of prediction results
            # will add 1 for inverse of data loading logic.
            if self.test_mode:
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
            else:
                bbox = [
                    int(float(bnd_box.find('xmin').text)) - 1,
                    int(float(bnd_box.find('ymin').text)) - 1,
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
            ignore = False
            if self.min_bbox_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_bbox_size or h < self.min_bbox_size:
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
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info

    def _filter_imgs(self, min_size=32, min_bbox_area_filter=None):
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area_filter (int | None): Filter images with bbox whose
                area smaller `min_bbox_area_filter`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indexes of `data_infos`.
        """
        valid_inds = []
        if min_bbox_area_filter is None:
            min_bbox_area_filter = self.min_bbox_area_filter
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int).tolist()
                if len(cat_ids) == 0:
                    continue
            if min_bbox_area_filter is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area_filter:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 class_splits=None):
        """Evaluate the predictions results in VOC protocol, and support to
        return evaluate results of specific categories.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original mmdet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consisent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.
        for i in range(len(results)):
            for j in range(len(results[i])):
                results[i][j][:, 0] += 1
                results[i][j][:, 1] += 1

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc07',
                    logger=logger,
                    tpfp_fn=voc_tpfp_fn)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[f'{k}: AP{int(iou_thr * 100):02d}'] = \
                            round(class_splits_mean_ap, 3)

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / \
                          len(class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


@DATASETS.register_module()
class FewShotVOCCopyDataset(FewShotVOCDataset):
    """For some meta learning method, the random sampled sampled support data
    is required for evaluation.

    FewShotVOCCopyDataset allow copy
    `data_infos` of other dataset by dumping `data_infos` into 'ann_cfg'.
    For example: ann_cfg = [dict(data_infos=FewShotVOCDataset.data_infos)]
    """

    def __init__(self, **kwargs):
        super(FewShotVOCCopyDataset, self).__init__(**kwargs)

    def ann_cfg_parser(self, ann_cfg):
        """Parse annotation config from a copy of other dataset.

        Args:
            ann_cfg (list[dict] | dict): contain other data_infos from dataset.
                Example: [dict(data_infos=FewShotVOCDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                f'{self.dataset_name}: ann_cfg of ' \
                f'FewShotVOCCopyDataset require data_infos.'
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    f'{self.dataset_name}: ann_cfg of ' \
                    f'FewShotVOCCopyDataset require data_infos.'
                data_infos += ann_cfg_['data_infos']
        return data_infos


@DATASETS.register_module()
class FewShotVOCDefaultDataset(FewShotVOCDataset):
    """FewShotVOCDefaultDataset provide predefine VOC annotation file for model
    reproducibility.

    The predefine annotation file provide fixed training data
    to avoid random sample few shot data. The `ann_cfg' should contain method
    and setting. For example: ann_cfg = [dict(method='TFA',
    setting='SPILT1_1shot')].
    """

    # predefined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA={
            f'SPLIT{split}_{shot}SHOT': [
                dict(
                    type='ann_file',
                    ann_file=f'data/few_shot_voc_split/{shot}shot/'
                    f'box_{shot}shot_{class_name}_train.txt',
                    ann_classes=[class_name])
                for class_name in VOC_SPLIT[f'ALL_CLASSES_SPLIT{split}']
            ]
            for shot in [1, 2, 3, 5, 10] for split in [1, 2, 3]
        },
        FSCE={
            f'SPLIT{split}_{shot}SHOT': [
                dict(
                    type='ann_file',
                    ann_file=f'data/few_shot_voc_split/{shot}shot/'
                    f'box_{shot}shot_{class_name}_train.txt',
                    ann_classes=[class_name])
                for class_name in VOC_SPLIT[f'ALL_CLASSES_SPLIT{split}']
            ]
            for shot in [1, 2, 3, 5, 10] for split in [1, 2, 3]
        },
        Attention_RPN={
            f'SPLIT{split}_{shot}SHOT': [
                dict(
                    type='ann_file',
                    ann_file=f'data/few_shot_voc_split/{shot}shot/'
                    f'box_{shot}shot_{class_name}_train.txt',
                    ann_classes=[class_name])
                for class_name in VOC_SPLIT[f'ALL_CLASSES_SPLIT{split}']
            ]
            for shot in [1, 2, 3, 5, 10] for split in [1, 2, 3]
        })

    def __init__(self, **kwargs):
        super(FewShotVOCDefaultDataset, self).__init__(**kwargs)

    def ann_cfg_parser(self, ann_cfg):
        """Parse predefine annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): contain method and setting
                of predefined annotation config. Example:
                [dict(method='TFA', setting='SPILT1_1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name}: ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = \
                        osp.join(ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        return super(FewShotVOCDataset, self).ann_cfg_parser(new_ann_cfg)
