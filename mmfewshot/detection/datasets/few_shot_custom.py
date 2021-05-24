import copy
import os.path as osp
import warnings

import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module()
class FewShotCustomDataset(CustomDataset):
    """Custom dataset for few shot detection.

    It allow single (normal dataset of fully supervised setting) or
    two (query-support fashion) pipelines for data processing.
    When annotation shots filter is used, it make sure accessible
    annotations meet the few shot setting in exact number of instances.

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
        ann_file (str): Annotation file path.
        pipeline (list[dict] | dict): Processing pipeline
            If is list[dict] all data will pass through this pipeline,
            If is dict, query and support data will be processed with
            two different pipelines and the dict should contain two keys:
                - 'query': list[dict]
                - 'support': list[dict]
        classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
        ann_shot_filter (dict, optional): If set None, all annotation from
            ann file will be loaded. If not None, annotation shot filter will
            specific which class and the maximum number of instances to load
            from annotation file. For example: {'dog': 10, 'person': 5}.
            Default: None.
    """

    CLASSES = None

    def __init__(
        self,
        ann_file,
        pipeline,
        classes,
        data_root=None,
        img_prefix='',
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        ann_shot_filter=None,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        self.ann_shot_filter = ann_shot_filter
        if self.ann_shot_filter is not None:
            for class_name in list(self.ann_shot_filter.keys()):
                assert class_name in self.CLASSES, \
                    f'class {class_name} from ' \
                    f'ann_shot_filter not in CLASSES, '

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        # filter annotations according to ann_shot_filter
        if self.ann_shot_filter is not None:
            self.data_infos = self._filter_annotations(self.data_infos,
                                                       self.ann_shot_filter)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline if there are two pipeline the
        # pipeline will be determined by key name of query or support
        if isinstance(pipeline, dict):
            self.pipeline = {}
            for key in pipeline.keys():
                self.pipeline[key] = Compose(pipeline[key])
        else:
            self.pipeline = Compose(pipeline)

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
            return self.pipeline[pipeline_key](results)

    def _filter_annotations(self, data_infos, ann_shot_filter):
        """Filter out annotations not in class_masks and excess annotations of
        specific class, while annotations of other classes in class_masks
        remain unchanged.

        Args:
            data_infos (list[dict]): Annotation infos.
            ann_shot_filter (dict): Specific which class and how many
            instances of each class to load from annotation file.
            For example: {'dog': 10, 'cat': 10, 'person': 5} Default: None.

        Returns:
            list[dict]: Annotation infos where number of specified class
                shots less than or equal to predefined number.
        """
        # build instance indexes of (img_id, gt_idx)
        total_instance_dict = {key: [] for key in ann_shot_filter.keys()}

        for data_info in data_infos:
            img_id = data_info['id']
            ann = data_info['ann']
            for i in range(ann['labels'].shape[0]):
                instance_class_name = self.CLASSES[ann['labels'][i]]
                if instance_class_name in ann_shot_filter.keys():
                    total_instance_dict[instance_class_name].append(
                        (img_id, i))

        total_instance_indexes = []
        for class_name in ann_shot_filter.keys():
            num_shot = ann_shot_filter[class_name]
            instance_indexes = total_instance_dict[class_name]
            # we will random sample from all instances to get exact
            # number of instance
            if len(instance_indexes) > num_shot:
                random_select = np.random.choice(
                    len(instance_indexes), num_shot, replace=False)
                total_instance_indexes += \
                    [instance_indexes[i] for i in random_select]
            # number of shot less than the predefined number,
            # which may cause the performance degradation
            elif len(instance_indexes) < num_shot:
                warning = f'number of {class_name} instance ' \
                          f'is {len(instance_indexes)} which is ' \
                          f'less than predefined shots {num_shot}.'
                warnings.warn(warning)
                total_instance_indexes += instance_indexes
            else:
                total_instance_indexes += instance_indexes

        new_data_infos = []
        for data_info in data_infos:
            img_id = data_info['id']
            selected_instance_index = \
                sorted([instance[1] for instance in total_instance_indexes
                        if instance[0] == img_id])
            ann = data_info['ann']
            if len(selected_instance_index) == 0:
                continue
            selected_ann = dict(
                bboxes=ann['bboxes'][selected_instance_index],
                labels=ann['labels'][selected_instance_index],
            )
            if ann.get('bboxes_ignore') is not None:
                selected_ann['bboxes_ignore'] = ann['bboxes_ignore']
            if ann.get('labels_ignore') is not None:
                selected_ann['labels_ignore'] = ann['labels_ignore']
            new_data_infos.append(
                dict(
                    id=img_id,
                    filename=data_info['filename'],
                    width=data_info['width'],
                    height=data_info['height'],
                    ann=selected_ann))
        return new_data_infos
