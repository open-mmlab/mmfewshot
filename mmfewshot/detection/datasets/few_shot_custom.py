import copy
import json
import os.path as osp
import warnings

import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
from mmdet.utils import get_root_logger
from terminaltables import AsciiTable

from .utils import NumpyEncoder


@DATASETS.register_module()
class FewShotCustomDataset(CustomDataset):
    """Custom dataset for few shot detection.

    The main differences with normal detection dataset fall in two aspects.

        - It allows to specify single (used in normal dataset) or multiple
            (used in query-support dataset) pipelines for data processing.
        - It supports to control the maximum number of instances of each class
            when loading the annotation file.

    The annotation format is shown as follows. The `ann` field
    is optional for testing.

    .. code-block:: none

        [
            {
                'id': '0000001'
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_cfg (list[dict]): Annotation config support two type of config.

            - loading annotation from common ann_file of dataset
              with or without specific classes.
              example:dict(type='ann_file', ann_file='path/to/ann_file',
              ann_classes=['dog', 'cat'])
            - loading annotation from a json file saved by dataset.
              example:dict(type='saved_dataset', ann_file='path/to/ann_file')
        classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class.
        pipeline (list[dict]): Config to specify processing pipeline.
            Used in normal dataset. Default: None.
        multi_pipelines (dict[list[dict]]): Config to specify
            data pipelines for corresponding data flow.
            For example, query and support data
            can be processed with two different pipelines, the dict
            should contain two keys like:

                - query (list[dict]): Config for query-data
                  process pipeline.
                - support (list[dict]): Config for support-data
                  process pipeline.
        data_root (str | None): Data root for ``ann_cfg``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
        filter_empty_gt (bool): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Default: Ture.
        min_bbox_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_bbox_size``, it would be added to ignored field.
            Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, all annotation from ann file would be loaded.
            Default: None.
        instance_wise (bool): If set true, `self.data_infos`
            would change to instance-wise, which means if the annotation
            of single image has more than one instance, the annotation would be
            split to num_instances items. Often used in support datasets,
            Default: False.
        dataset_name (str | None): Name of dataset to display. For example:
            'train_dataset' or 'query_dataset'. Default: None.
    """

    CLASSES = None

    def __init__(self,
                 ann_cfg,
                 classes,
                 pipeline=None,
                 multi_pipelines=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 min_bbox_size=None,
                 ann_shot_filter=None,
                 instance_wise=False,
                 dataset_name=None):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        if classes is not None:
            self.CLASSES = self.get_classes(classes)
        self.instance_wise = instance_wise
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        self.ann_cfg = copy.deepcopy(ann_cfg)
        self.data_infos = self.ann_cfg_parser(ann_cfg)

        assert self.data_infos is not None, \
            f'{self.dataset_name} : none annotation loaded.'

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            if min_bbox_size:
                self.data_infos = self._filter_bboxs(min_bbox_size)
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # filter annotations by ann_shot_filter
            if ann_shot_filter is not None:
                if isinstance(ann_shot_filter, dict):
                    for class_name in list(ann_shot_filter.keys()):
                        assert class_name in self.CLASSES, \
                            f'{self.dataset_name} : class ' \
                            f'{class_name} in ann_shot_filter not in CLASSES.'
                elif isinstance(ann_shot_filter, int):
                    ann_shot_filter = {
                        class_name: ann_shot_filter
                        for class_name in self.CLASSES
                    }
                else:
                    raise ValueError(
                        'ann_shot_filter only support dict or int')
                self.ann_shot_filter = ann_shot_filter
                self.data_infos = self._filter_annotations(
                    self.data_infos, self.ann_shot_filter)
            # instance_wise will make each data info only contain one
            # annotation otherwise all annotation from same image will
            # be checked and merged.
            if self.instance_wise:
                instance_wise_data_infos = []
                for data_info in self.data_infos:
                    num_instance = data_info['ann']['labels'].size
                    if num_instance > 1:
                        for i in range(data_info['ann']['labels'].size):
                            tmp_data_info = copy.deepcopy(data_info)
                            tmp_data_info['ann']['labels'] = np.expand_dims(
                                data_info['ann']['labels'][i], axis=0)
                            tmp_data_info['ann']['bboxes'] = np.expand_dims(
                                data_info['ann']['bboxes'][i, :], axis=0)
                            instance_wise_data_infos.append(tmp_data_info)
                    else:
                        instance_wise_data_infos.append(data_info)
                self.data_infos = instance_wise_data_infos
            else:
                merge_data_dict = {}
                for i, data_info in enumerate(self.data_infos):
                    if merge_data_dict.get(data_info['id'], None) is None:
                        merge_data_dict[data_info['id']] = data_info
                    else:
                        ann_a = merge_data_dict[data_info['id']]['ann']
                        ann_b = data_info['ann']
                        merge_dat_info = {
                            'bboxes':
                            np.concatenate((ann_a['bboxes'], ann_b['bboxes'])),
                            'labels':
                            np.concatenate((ann_a['labels'], ann_b['labels'])),
                        }
                        if ann_a.get('bboxes_ignore', None) is not None:
                            if not (ann_a['bboxes_ignore']
                                    == ann_b['bboxes_ignore']).all():
                                merge_dat_info['bboxes_ignore'] = \
                                    np.concatenate((ann_a['bboxes_ignore'],
                                                    ann_b['bboxes_ignore']))
                                merge_dat_info['labels_ignore'] = \
                                    np.concatenate((ann_a['labels_ignore'],
                                                    ann_b['labels_ignore']))
                        merge_data_dict[
                            data_info['id']]['ann'] = merge_dat_info
                self.data_infos = [
                    merge_data_dict[key] for key in merge_data_dict.keys()
                ]
            # set group flag for the sampler
            self._set_group_flag()

        assert pipeline is None or multi_pipelines is None, \
            f'{self.dataset_name} : can not assign pipeline ' \
            f'or multi_pipelines simultaneously'
        # processing pipeline if there are two pipeline the
        # pipeline will be determined by key name of query or support
        if multi_pipelines is not None:
            assert isinstance(multi_pipelines, dict), \
                f'{self.dataset_name} : multi_pipelines is type of dict'
            self.multi_pipelines = {}
            for key in multi_pipelines.keys():
                self.multi_pipelines[key] = Compose(multi_pipelines[key])
        elif pipeline is not None:
            assert isinstance(pipeline, list), \
                f'{self.dataset_name} : pipeline is type of list'
            self.pipeline = Compose(pipeline)
        else:
            raise ValueError('missing pipeline or multi_pipelines')

        # show dataset annotation usage
        logger = get_root_logger()
        logger.info(self.__repr__())

    def ann_cfg_parser(self, ann_cfg):
        """Parse annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): Annotation config support two type of config.

                - 'ann_file': loading annotation from common ann_file of
                    dataset. example: dict(type='ann_file',
                    ann_file='path/to/ann_file', ann_classes=['dog', 'cat'])
                - 'saved_dataset': loading annotation from saved dataset.
                    example:dict(type='saved_dataset',
                    ann_file='path/to/ann_file')

        Returns:
            list[dict]: Annotation information.
        """
        if self.data_root is not None:
            for i in range(len(ann_cfg)):
                if not osp.isabs(ann_cfg[i]['ann_file']):
                    ann_cfg[i]['ann_file'] = \
                        osp.join(self.data_root, ann_cfg[i]['ann_file'])
        # Predefined ann_cfg must be list
        assert isinstance(ann_cfg, list), \
            f'{self.dataset_name} : ann_cfg should be type of list.'
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name} : ann_cfg should be list of dict.'
            assert ann_cfg_['type'] in ['ann_file', 'saved_dataset'], \
                f'{self.dataset_name} : ann_cfg only support type of ' \
                f'ann_file and saved_dataset'
        return self.load_annotations(ann_cfg)

    def get_ann_info(self, idx):
        """Get annotation by index.

        When override this function please make sure same annotations are used
        during the whole training.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return copy.deepcopy(self.data_infos[idx]['ann'])

    def prepare_train_img(self, idx, pipeline_key=None, gt_idx=None):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.
            pipeline_key (str): Name of pipeline
            gt_idx (list[int]): Index of used annotation.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        # annotation filter
        if gt_idx is not None:
            selected_ann_info = {
                'bboxes': ann_info['bboxes'][gt_idx],
                'labels': ann_info['labels'][gt_idx]
            }
            # keep pace with new annotations
            new_img_info = copy.deepcopy(img_info)
            new_img_info['ann'] = selected_ann_info
            results = dict(img_info=new_img_info, ann_info=selected_ann_info)
        else:
            results = dict(img_info=copy.deepcopy(img_info), ann_info=ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        self.pre_pipeline(results)
        if pipeline_key is None:
            return self.pipeline(results)
        else:
            return self.multi_pipelines[pipeline_key](results)

    def _filter_annotations(self, data_infos, ann_shot_filter):
        """Filter out extra annotations of specific class, while annotations of
        classes not in filter remain unchanged and the ignored annotations will
        be removed.

        Args:
            data_infos (list[dict]): Annotation infos.
            ann_shot_filter (dict): Specific which class and how many
                instances of each class to load from annotation file.
                For example: {'dog': 10, 'cat': 10, 'person': 5}

        Returns:
            list[dict]: Annotation infos where number of specified class
                shots less than or equal to predefined number.
        """
        if ann_shot_filter is None:
            return data_infos
        # build instance indexes of (img_id, gt_idx)
        filter_instances = {key: [] for key in ann_shot_filter.keys()}
        keep_instances_indexes = []
        for idx, data_info in enumerate(data_infos):
            ann = data_info['ann']
            for i in range(ann['labels'].shape[0]):
                instance_class_name = self.CLASSES[ann['labels'][i]]
                if instance_class_name in ann_shot_filter.keys():
                    filter_instances[instance_class_name].append((idx, i))
                else:
                    keep_instances_indexes.append((idx, i))
        # filter extra shots
        for class_name in ann_shot_filter.keys():
            num_shots = ann_shot_filter[class_name]
            instance_indexes = filter_instances[class_name]
            # random sample from all instances
            if len(instance_indexes) > num_shots:
                random_select = np.random.choice(
                    len(instance_indexes), num_shots, replace=False)
                keep_instances_indexes += \
                    [instance_indexes[i] for i in random_select]
            # number of available shots less than the predefined number,
            # which may cause the performance degradation
            else:
                if len(instance_indexes) < num_shots:
                    warnings.warn(f'number of {class_name} instance is '
                                  f'{len(instance_indexes)} which is '
                                  f'less than predefined shots {num_shots}.')
                keep_instances_indexes += instance_indexes

        new_data_infos = []
        for idx, data_info in enumerate(data_infos):
            selected_instance_index = \
                sorted([instance[1] for instance in keep_instances_indexes
                        if instance[0] == idx])
            if len(selected_instance_index) == 0:
                continue
            ann = data_info['ann']
            selected_ann = dict(
                bboxes=ann['bboxes'][selected_instance_index],
                labels=ann['labels'][selected_instance_index],
            )
            new_data_infos.append(
                dict(
                    id=data_info['id'],
                    filename=data_info['filename'],
                    width=data_info['width'],
                    height=data_info['height'],
                    ann=selected_ann))
        return new_data_infos

    def _filter_bboxs(self, min_bbox_size):
        new_data_infos = []
        for data_info in self.data_infos:
            ann = data_info['ann']
            keep_idx, ignore_idx = [], []
            for i in range(ann['bboxes'].shape[0]):
                bbox = ann['bboxes'][i]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < min_bbox_size or h < min_bbox_size:
                    ignore_idx.append(i)
                else:
                    keep_idx.append(i)
            if len(ignore_idx) > 0:
                bboxes_ignore = ann.get('bboxes_ignore', np.zeros((0, 4)))
                labels_ignore = ann.get('labels_ignore', np.zeros((0, )))
                new_bboxes_ignore = ann['bboxes'][ignore_idx]
                new_labels_ignore = ann['labels'][ignore_idx]
                bboxes_ignore = np.concatenate(
                    (bboxes_ignore, new_bboxes_ignore))
                labels_ignore = np.concatenate(
                    (labels_ignore, new_labels_ignore))
                data_info.update(
                    ann=dict(
                        bboxes=ann['bboxes'][keep_idx],
                        labels=ann['labels'][keep_idx],
                        bboxes_ignore=bboxes_ignore,
                        labels_ignore=labels_ignore))
            new_data_infos.append(data_info)
        return new_data_infos

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In few shot setting, the limited number of images
        might cause some mini-batch always sample a certain number of images
        and thus not fully shuffle the data, which may degrade the performance.
        Therefore, all flags are simply set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations_saved(self, ann_cfg):
        """Load data_infos from saved json."""
        with open(ann_cfg) as f:
            data_infos = json.load(f)
        meta_idx = None
        for i, data_info in enumerate(data_infos):
            if 'CLASSES' in data_info.keys():
                assert self.CLASSES == tuple(data_info['CLASSES']), \
                    f'{self.dataset_name} : class labels mismatch.'
                assert self.img_prefix == data_info['img_prefix'], \
                    f'{self.dataset_name} : image prefix mismatch.'
                meta_idx = i
                continue
            for k in data_info['ann']:
                if isinstance(data_info['ann'][k], list):
                    if len(data_info['ann'][k]) == 0 and k == 'bboxes_ignore':
                        data_info['ann'][k] = np.zeros((0, 4))
                    else:
                        data_info['ann'][k] = np.array(data_info['ann'][k])
                    if 'box' in k:
                        data_info['ann'][k] = data_info['ann'][k].astype(
                            np.float32)
                    else:
                        data_info['ann'][k] = data_info['ann'][k].astype(
                            np.int64)
        if meta_idx is not None:
            data_infos.pop(meta_idx)
        return data_infos

    def save_data_infos(self, output_path):
        """Save data_infos into json."""
        meta_info = [{'CLASSES': self.CLASSES, 'img_prefix': self.img_prefix}]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                meta_info + self.data_infos,
                f,
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder)

    def __repr__(self):
        """Print the number of instances of each class."""
        result = (f'\n{self.__class__.__name__} {self.dataset_name} '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []

        table = AsciiTable(table_data)
        result += table.table
        return result
