import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from terminaltables import AsciiTable

from .few_shot_custom import FewShotCustomDataset

# predefined classes split for few shot setting
COCO_SPLIT = dict(
    ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    NOVEL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'bottle', 'chair', 'couch', 'potted plant',
                   'dining table', 'tv'),
    BASE_CLASSES=('truck', 'traffic light', 'fire hydrant', 'stop sign',
                  'parking meter', 'bench', 'elephant', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush'))


@DATASETS.register_module()
class FewShotCocoDataset(FewShotCustomDataset):
    """COCO dataset for few shot detection.

    Args:
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load predefined classes in :obj:`FewShotCocoDataset`.
            For example: 'BASE_CLASSES', 'NOVEL_CLASSES` or `ALL_CLASSES`.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used for each base
            class. If is None, all annotation will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
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
                 min_bbox_area=None,
                 dataset_name=None,
                 test_mode=False,
                 **kwargs):
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.SPLIT = COCO_SPLIT
        assert classes is not None, f'{self.dataset_name} : classes ' \
                                    f'in FewShotCocoDataset can not be None.'
        # configure ann_shot_filter by num_novel_shots and num_base_shots
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes)
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is not None or num_base_shots is not None, \
                f'{self.dataset_name} : can not config ann_shot_filter ' \
                f'and num_novel_shots/num_base_shots at the same time.'

        # these values would be set in `self.load_annotations_coco`
        self.cat_ids = []
        self.cat2label = {}
        self.coco = None
        self.img_ids = None

        super(FewShotCocoDataset, self).__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def get_classes(self, classes):
        """Get class names.

        Args:
            classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load predefined classes in `FewShotCocoDataset`.
            For example: 'NOVEL_CLASSES'.

        Returns:
            list[str]: list of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(), \
                f'{self.dataset_name} : not a predefine classes or ' \
                f'split in COCO_SPLIT.'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name} : BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name} : NOVEL_CLASSES do not have ' \
                    f'base instances.'
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self):
        """generate ann_shot_filter by novel and base classes."""
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT['NOVEL_CLASSES']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT['BASE_CLASSES']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg):
        """Support to Load annotation from two type of ann_cfg.

            - type of 'ann_file': COCO-style annotation file.
            - type of 'saved_dataset': Saved COCO dataset json.

        Args:
            ann_cfg (list[dict]): Config of annotations.

        Returns:
            list[dict]: Annotation infos.
        """
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                data_infos += self.load_annotations_coco(ann_cfg_['ann_file'])
            else:
                raise ValueError(f'not support annotation type '
                                 f'{ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_coco(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # to keep the label order equal to the order in CLASSES
        for i, class_name in enumerate(self.CLASSES):
            cat_id = self.coco.get_cat_ids(cat_names=[class_name])[0]
            self.cat_ids.append(cat_id)
            self.cat2label[cat_id] = i
        self.img_ids = self.coco.get_img_ids()

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['ann'] = self._get_ann_info(info)
            if 'train2014' in info['filename']:
                info['filename'] = 'train2014/' + info['filename']
            elif 'val2014' in info['filename']:
                info['filename'] = 'val2014/' + info['filename']
            elif 'instances_val2017' in ann_file:
                info['filename'] = 'val2017/' + info['filename']
            elif 'instances_train2017' in ann_file:
                info['filename'] = 'train2017/' + info['filename']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), \
            f'{self.dataset_name} : Annotation ids in {ann_file} ' \
            f'are not unique!'
        return data_infos

    def _get_ann_info(self, data_info):
        """Get COCO annotation by index.

        Args:
            data_info(dict): Data info.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = data_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(data_info, ann_info)

    def _filter_imgs(self, min_size=32, min_bbox_area=None):
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indexes of data_infos.
        """
        valid_inds = []
        valid_img_ids = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and img_info['ann']['labels'].size == 0:
                continue
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
            valid_img_ids.append(img_info['id'])
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str, str]: Possible keys are "bbox", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | np.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json file paths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 class_splits=None):
        """Evaluation in COCO protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'proposal', 'proposal_fast'. Default: 'bbox'
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'``.
            class_splits: (list[str] | None): Calculate metric of classes split
                in COCO_SPLIT. For example: ['BASE_CLASSES', 'NOVEL_CLASSES'].
                Default: None.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'please define classes split.'
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            # eval each class splits
            if class_splits is not None:
                class_splits = {k: COCO_SPLIT[k] for k in class_splits}
                for split_name in class_splits.keys():
                    split_cat_ids = [
                        self.cat_ids[i] for i in range(len(self.CLASSES))
                        if self.CLASSES[i] in class_splits[split_name]
                    ]
                    self._evaluate_by_class_split(
                        cocoGt,
                        cocoDt,
                        iou_type,
                        proposal_nums,
                        iou_thrs,
                        split_cat_ids,
                        metric,
                        metric_items,
                        eval_results,
                        False,
                        logger,
                        split_name=split_name + ' ')
            # eval all classes
            self._evaluate_by_class_split(cocoGt, cocoDt, iou_type,
                                          proposal_nums, iou_thrs,
                                          self.cat_ids, metric, metric_items,
                                          eval_results, classwise, logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def _evaluate_by_class_split(self,
                                 cocoGt,
                                 cocoDt,
                                 iou_type,
                                 proposal_nums,
                                 iou_thrs,
                                 cat_ids,
                                 metric,
                                 metric_items,
                                 eval_results,
                                 classwise,
                                 logger,
                                 split_name=''):
        """Evaluation a split of classes in COCO protocol.

        Args:
            cocoGt (object): coco object with ground truth annotations.
            cocoDt (object): coco object with detection results.
            iou_type (str): Type of IOU.
            proposal_nums (int): Number of proposals.
            iou_thrs (list[int]): Thresholds of IoU.
            cat_ids (list[int]): Class ids of classes to be evaluated.
            metric (str): Metrics to be evaluated.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'``.
            eval_results dict[str, float]: COCO style evaluation metric.
            classwise (bool): Whether to evaluating the AP for each class.
            split_name (str): Name of split. Default:''.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.params.catIds = cat_ids
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')
        if split_name is not None:
            print_log(
                f'###################################'
                f'\n evaluation of {split_name} class',
                logger=logger)
        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]

            for item in metric_items:
                val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[split_name + item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2], \
                    f'{self.cat_ids},{precisions.shape}'

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [split_name + 'category', split_name + 'AP'] * (
                    num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                eval_results[split_name + key] = val
            ap = cocoEval.stats[:6]
            eval_results[split_name + f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

            return eval_results


@DATASETS.register_module()
class FewShotCocoCopyDataset(FewShotCocoDataset):
    """Only used in evaluation of some meta-learning method.

    For some meta-learning methods, the random sampled support data in the
    training phase is required for evaluation. The usage of `ann_cfg` is
    different from :obj:`FewShotCocoDataset`. :obj:`FewShotCocoCopyDataset`
    support to load `data_infos` of other datasets via `ann_cfg`.

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotCocoDataset.data_infos)]
    """

    def __init__(self, ann_cfg, **kwargs):
        super(FewShotCocoCopyDataset, self).__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg):
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotCocoDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                'ann_cfg of FewShotCocoCopyDataset require data_infos.'
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    'ann_cfg of FewShotCocoCopyDataset require data_infos.'
                data_infos += ann_cfg_['data_infos']
        return data_infos


@DATASETS.register_module()
class FewShotCocoDefaultDataset(FewShotCocoDataset):
    """FewShot COCO Dataset with some predefine annotation paths.

    :obj:`FewShotCocoDefaultDataset` provides predefine annotation files
    to ensure the reproducibility. The predefine annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotCocoDataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='TFA', setting='1shot')].
    """
    # predefined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA={
            f'{shot}SHOT': [
                dict(
                    type='ann_file',
                    ann_file=f'data/few_shot_coco_split/{shot}shot/'
                    f'full_box_{shot}shot_{class_name}_trainval.json')
                for class_name in COCO_SPLIT['ALL_CLASSES']
            ]
            for shot in [10, 30]
        },
        FSCE={
            f'{shot}SHOT': [
                dict(
                    type='ann_file',
                    ann_file=f'data/few_shot_coco_split/{shot}shot/'
                    f'full_box_{shot}shot_{class_name}_trainval.json')
                for class_name in COCO_SPLIT['ALL_CLASSES']
            ]
            for shot in [10, 30]
        },
        Attention_RPN={
            '10SHOT': [
                dict(
                    type='ann_file',
                    ann_file='data/few_shot_coco_split/fsod_10shot/'
                    'final_split_voc_10_shot_instances_train2017.json')
            ],
            '30SHOT': [
                dict(
                    type='ann_file',
                    ann_file=f'data/few_shot_coco_split/'
                    f'30shot/full_box_30shot_{class_name}_trainval.json')
                for class_name in COCO_SPLIT['ALL_CLASSES']
            ]
        })

    def __init__(self, ann_cfg, **kwargs):
        super(FewShotCocoDefaultDataset, self).__init__(
            ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg):
        """Parse predefine annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): Each dict should contain
                `method` and `setting` to get corresponding
                annotation from `DEFAULT_ANN_CONFIG`.
                For example: [dict(method='TFA', setting='1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name} : ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = \
                        osp.join(ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        return super(FewShotCocoDataset, self).ann_cfg_parser(new_ann_cfg)
