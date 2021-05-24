import copy
import warnings

import numpy as np
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class MergeDataset(object):
    """A wrapper of merge dataset.

    This dataset wrapper would be called when using multiple annotation
    files for NwayKshotDataset, QueryAwareDataset, and FewShotCustomDataset.
    It would merge the data info of input datasets, because different
    annotations of same image will cross different datasets.


    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        self.dataset = copy.deepcopy(datasets[0])
        self.CLASSES = self.dataset.CLASSES
        for dataset in datasets:
            assert dataset.img_prefix == self.dataset.img_prefix, \
                'when using MergeDataset all img_prefix should be the same'

        self.img_prefix = self.dataset.img_prefix

        # merge datainfos for all datasets
        concat_data_infos = sum([dataset.data_infos for dataset in datasets],
                                [])
        merge_data_dict = {}
        for i, data_info in enumerate(concat_data_infos):

            if merge_data_dict.get(data_info['id'], None) is None:
                merge_data_dict[data_info['id']] = data_info
            else:
                merge_data_dict[data_info['id']]['ann'] = \
                    self.merge_ann(merge_data_dict[data_info['id']]['ann'],
                                   data_info['ann'])

        self.dataset.data_infos = [
            merge_data_dict[key] for key in merge_data_dict.keys()
        ]

        # Disable the groupsampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(datasets[0], 'flag'):
            self.flag = np.zeros(len(self.dataset), dtype=np.uint8)

    def get_cat_ids(self, idx):
        """Get category ids of merge dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return self.dataset.get_cat_ids(idx)

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
        return self.dataset.prepare_train_img(idx, pipeline_key, gt_idx)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.dataset.get_ann_info(idx)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        """Dataset length after merge."""
        return len(self.dataset)

    def __repr__(self):
        return self.dataset.__repr__()

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        eval_results = self.dataset.evaluate(results, logger=logger, **kwargs)
        return eval_results

    @staticmethod
    def merge_ann(ann_a, ann_b):
        """Merge two annotations.

        Args:
            ann_a (dict): Dict of annotation.
            ann_b (dict): Dict of annotation.

        Returns:
            dict: Merged annotation.
        """
        assert sorted(ann_a.keys()) == sorted(ann_b.keys()), \
            'can not merge different type of annotations'
        return {
            'bboxes': np.concatenate((ann_a['bboxes'], ann_b['bboxes'])),
            'labels': np.concatenate((ann_a['labels'], ann_b['labels'])),
            'bboxes_ignore': ann_a['bboxes_ignore'],
            'labels_ignore': ann_a['labels_ignore']
        }


@DATASETS.register_module()
class QueryAwareDataset(object):
    """A wrapper of query aware dataset.

    For each item in dataset, there will be one query image and
    (num_support_way * num_support_shot) support images.
    The support images are sampled according to the selected
    query image and include positive class (random one class
    in query image) and negative class (any classes not appear in
     query image).

    Args:
        datasets (obj:`FewShotDataset`, `MergeDataset`):
            The dataset to be wrapped.
        num_support_way (int): The number of classes for support data,
            the first one always be the positive class.
        num_support_shot (int): The number of shot for each support class.
    """

    def __init__(self, dataset, num_support_way, num_support_shot):
        self.dataset = dataset
        self.num_support_way = num_support_way
        self.num_support_shot = num_support_shot
        self.CLASSES = dataset.CLASSES
        assert self.num_support_way <= len(self.CLASSES), \
            'Please set the num_support_way smaller than the ' \
            'number of classes.'
        # build data index (idx, gt_idx) by class.
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        # count max number of anns in one image for each class, which will
        # decide whether sample repeated instance or not.
        self.max_anns_per_image_by_class = [
            0 for _ in range(len(self.CLASSES))
        ]
        # count image for each class annotation when novel class only
        # has one image, the positive support is allowed sampled from itself.
        self.num_image_by_class = [0 for _ in range(len(self.CLASSES))]

        for idx in range(len(self.dataset)):
            labels = self.dataset.get_ann_info(idx)['labels']
            class_count = [0 for _ in range(len(self.CLASSES))]
            for gt_idx, gt in enumerate(labels):
                self.data_infos_by_class[gt].append((idx, gt_idx))
                class_count[gt] += 1
            for i in range(len(self.CLASSES)):
                # number of images for each class
                if class_count[i] > 0:
                    self.num_image_by_class[i] += 1
                # max number of one class annotations in one image
                if class_count[i] > self.max_anns_per_image_by_class[i]:
                    self.max_anns_per_image_by_class[i] = class_count[i]

        for i in range(len(self.CLASSES)):
            assert len(self.data_infos_by_class[i]) > 0, \
                f'Class {self.CLASSES[i]} has zero annotation'
            if len(self.data_infos_by_class[i]) <= self.num_support_shot - \
                    self.max_anns_per_image_by_class[i]:
                warnings.warn(
                    f'During training, instances of class {self.CLASSES[i]} '
                    f'may smaller than the number of support shots which '
                    f'causes some instance will be sampled multiple times')
            if self.num_image_by_class[i] == 1:
                warnings.warn(f'Class {self.CLASSES[i]} only have one '
                              f'image, query and support will sample '
                              f'from instance of same image')

        # Disable the groupsampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(dataset, 'flag'):
            self.flag = np.zeros(len(self.dataset), dtype=np.uint8)

    def __getitem__(self, idx):
        # sample query data
        try_time = 0
        while True:
            try_time += 1
            cat_ids = self.dataset.get_cat_ids(idx)
            # query image have too many classes, can not find enough
            # negative support classes.
            if len(self.CLASSES) - len(cat_ids) >= self.num_support_way - 1:
                break
            else:
                idx = self._rand_another(idx)
            assert try_time < 100, \
                'Not enough negative support classes for query image,' \
                ' please try a smaller support way.'

        query_class = np.random.choice(cat_ids)
        query_gt_idx = [
            i for i in range(len(cat_ids)) if cat_ids[i] == query_class
        ]
        query_data = self.dataset.prepare_train_img(idx, 'query', query_gt_idx)
        query_data['query_class'] = [query_class]

        # sample negative support classes, which not appear in query image
        support_class = [
            i for i in range(len(self.CLASSES)) if i not in cat_ids
        ]
        support_class = np.random.choice(
            support_class,
            min(self.num_support_way - 1, len(support_class)),
            replace=False)
        support_idxes = self.generate_support(idx, query_class, support_class)
        support_data = [
            self.dataset.prepare_train_img(idx, 'support', [gt_idx])
            for (idx, gt_idx) in support_idxes
        ]
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self):
        """Length after repetition."""
        return len(self.dataset)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def generate_support(self, idx, query_class, support_classes):
        """Generate support indexes of query images.

        Args:
            idx (int): Index of query data.
            query_class (int): Query class.
            support_classes (list[int]): Classes of support data.

        Returns:
            list[(int, int)]: A batch (num_support_way * num_support_shot)
                of support data (idx, gt_idx).
        """
        support_idxes = []
        if self.num_image_by_class[query_class] == 1:
            # only have one image, instance will sample from same image
            pos_support_idxes = self.sample_support_shots(
                idx, query_class, allow_same_image=True)
        else:
            # instance will sample from different image from query image
            pos_support_idxes = self.sample_support_shots(idx, query_class)
        support_idxes.extend(pos_support_idxes)
        for support_class in support_classes:
            neg_support_idxes = self.sample_support_shots(idx, support_class)
            support_idxes.extend(neg_support_idxes)
        return support_idxes

    def sample_support_shots(self, idx, class_id, allow_same_image=False):
        """Generate positive support indexes by class id.

        Args:
            idx (int): Index of query data.
            class_id (int): Support class.
            allow_same_image: Allow instance sampled from same image
                as query image. Default: False.
        Returns:
            list[(int, int)]: Support data (num_support_shot)
                of specific class.
        """
        support_idxes = []
        num_total_shot = len(self.data_infos_by_class[class_id])
        num_ignore_shot = self.count_class_id(idx, class_id)
        # set num_sample_shots for each time of sampling

        if num_total_shot - num_ignore_shot < self.num_support_shot:
            # if not have enough support data allow repeated data
            num_sample_shots = num_total_shot
            allow_repeat = True
        else:
            # if have enough support data not allow repeated data
            num_sample_shots = self.num_support_shot
            allow_repeat = False
        while len(support_idxes) < self.num_support_shot:
            selected_gt_idxes = np.random.choice(
                num_total_shot, num_sample_shots, replace=False)

            selected_gts = [
                self.data_infos_by_class[class_id][selected_gt_idx]
                for selected_gt_idx in selected_gt_idxes
            ]
            for selected_gt in selected_gts:
                # filter out query annotations
                if selected_gt[0] == idx:
                    if not allow_same_image:
                        continue
                if allow_repeat:
                    support_idxes.append(selected_gt)
                elif selected_gt not in support_idxes:
                    support_idxes.append(selected_gt)
                if len(support_idxes) == self.num_support_shot:
                    break
            # update the number of data for next time sample
            num_sample_shots = min(self.num_support_shot - len(support_idxes),
                                   num_sample_shots)
        return support_idxes

    def count_class_id(self, idx, class_id):
        """Count number of instance of specific."""
        cat_ids = self.dataset.get_cat_ids(idx)
        cat_ids_of_class = [
            i for i in range(len(cat_ids)) if cat_ids[i] == class_id
        ]
        return len(cat_ids_of_class)


@DATASETS.register_module()
class NwayKshotDataset(object):
    """A dataset wrapper of NwayKshotDataset.

    Based on incoming dataset, query dataset will sample batch data as
    regular dataset, while support dataset will pre sample batch data
    indexes. Each batch index contain (num_support_way * num_support_shot)
    samples. The default format of NwayKshotDataset is query dataset and
    the query dataset will convert into support dataset by using convert
    function.

    Args:
        datasets (obj:`FewShotDataset`, `MergeDataset`):
            The dataset to be wrapped.
        num_support_way (int):
            The number of classes in support data batch.
        num_support_shot (int):
            The number of shots for each class in support data batch.
    """

    def __init__(self, dataset, num_support_way, num_support_shot):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        # The data_type determinate the behavior of fetching data,
        # the default data_type is 'query', which is the same as regular
        # dataset. To convert the dataset into 'support' dataset, simply
        # call the function convert_query_to_support().
        self.data_type = 'query'
        self.num_support_way = num_support_way
        assert num_support_way <= len(self.CLASSES), \
            'support way can not larger than the number of classes'
        self.num_support_shot = num_support_shot
        self.batch_index = []
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}

        # Disable the groupsampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(dataset, 'flag'):
            self.flag = np.zeros(len(self.dataset), dtype=np.uint8)

    def __getitem__(self, idx):
        if self.data_type == 'query':
            # loads one data in query pipeline
            return self.dataset.prepare_train_img(idx, 'query')
        elif self.data_type == 'support':
            # loads one batch of data in support pipeline
            b_idx = self.batch_index[idx]
            batch_data = [
                self.dataset.prepare_train_img(idx, 'support', [gt_idx])
                for (idx, gt_idx) in b_idx
            ]
            return batch_data
        else:
            raise ValueError('not support data type')

    def __len__(self):
        """Length of dataset."""
        if self.data_type == 'query':
            return len(self.dataset)
        elif self.data_type == 'support':
            return len(self.batch_index)
        else:
            raise ValueError('not support data type')

    def shuffle_support(self):
        """Generate new batch indexes."""
        if self.data_type == 'query':
            raise ValueError('not support data type')
        self.batch_index = self.generate_batch_index(len(self.batch_index))

    def convert_query_to_support(self, support_dataset_len):
        """Convert query dataset to support dataset.

        Args:
            support_dataset_len (int): Length of pre sample batch indexes.
        """
        # create lookup table for annotations in same class
        for idx in range(len(self.dataset)):
            labels = self.dataset.get_ann_info(idx)['labels']
            for gt_idx, gt in enumerate(labels):
                self.data_infos_by_class[gt].append((idx, gt_idx))
        # make sure all class index lists have enough
        # instances (length > num_support_shot)
        for i in range(len(self.CLASSES)):
            num_gts = len(self.data_infos_by_class[i])
            if num_gts < self.num_support_shot:
                self.data_infos_by_class[i] = self.data_infos_by_class[i] * \
                                        (self.num_support_shot // num_gts + 1)
        self.batch_index = self.generate_batch_index(support_dataset_len)
        self.data_type = 'support'
        if hasattr(self, 'flag'):
            self.flag = np.zeros(support_dataset_len, dtype=np.uint8)

    def generate_batch_index(self, dataset_len):
        """Generate batch index [length of datasets * [support way * support shots]].

        Args:
            dataset_len: Length of pre sample batch indexes.

        Returns:
            List[List[(data_idx, gt_idx)]]: Pre sample batch indexes.
        """
        total_batch_index = []
        for _ in range(dataset_len):
            batch_index = []
            selected_classes = np.random.choice(
                len(self.CLASSES), self.num_support_way, replace=False)
            for cls in selected_classes:
                num_gts = len(self.data_infos_by_class[cls])
                selected_gts_idx = np.random.choice(
                    num_gts, self.num_support_shot, replace=False)
                selected_gts = [
                    self.data_infos_by_class[cls][gt_idx]
                    for gt_idx in selected_gts_idx
                ]
                batch_index.extend(selected_gts)
            total_batch_index.append(batch_index)
        return total_batch_index
