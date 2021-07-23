import copy
import json
import warnings

import numpy as np
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class QueryAwareDataset(object):
    """A wrapper of query aware dataset.

    For each item in query aware dataset, there will be one query image and
    (num_support_ways * num_support_shots) support images. The support images
    are sampled according to the selected query image and include positive
    class (random classes in query image) and negative class (any classes not
    appear in query image).

    Args:
        query_dataset (obj:`FewShotCustomDataset`):
            Query dataset to be wrapped.
        support_dataset (obj:`FewShotCustomDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch, the first one always be the positive class.
        num_support_shots (int): Number of support shots for each
            class in mini-batch, the first K shots always from positive class.
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset,
                 support_dataset,
                 num_support_ways,
                 num_support_shots,
                 repeat_times=1):
        self.query_dataset = query_dataset
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        self.CLASSES = self.query_dataset.CLASSES
        self.repeat_times = repeat_times
        assert self.num_support_ways <= len(self.CLASSES), \
            'Please set the num_support_ways smaller than the ' \
            'number of classes.'
        # build data index (idx, gt_idx) by class.
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        # counting max number of anns in one image for each class, which will
        # decide whether sample repeated instance or not.
        self.max_anns_num_one_image = [0 for _ in range(len(self.CLASSES))]
        # count image for each class annotation when novel class only
        # has one image, the positive support is allowed sampled from itself.
        self.num_image_by_class = [0 for _ in range(len(self.CLASSES))]

        for idx in range(len(self.support_dataset)):
            labels = self.support_dataset.get_ann_info(idx)['labels']
            class_count = [0 for _ in range(len(self.CLASSES))]
            for gt_idx, gt in enumerate(labels):
                self.data_infos_by_class[gt].append((idx, gt_idx))
                class_count[gt] += 1
            for i in range(len(self.CLASSES)):
                # number of images for each class
                if class_count[i] > 0:
                    self.num_image_by_class[i] += 1
                # max number of one class annotations in one image
                if class_count[i] > self.max_anns_num_one_image[i]:
                    self.max_anns_num_one_image[i] = class_count[i]

        for i in range(len(self.CLASSES)):
            assert len(self.data_infos_by_class[i]) > 0, \
                f'Class {self.CLASSES[i]} has zero annotation'
            if len(self.data_infos_by_class[i]) <= \
                    self.num_support_shots - \
                    self.max_anns_num_one_image[i]:
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
        if hasattr(self.query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        self._ori_len = len(self.query_dataset)

    def __getitem__(self, idx):
        idx %= self._ori_len
        # sample query data
        try_time = 0
        while True:
            try_time += 1
            cat_ids = self.query_dataset.get_cat_ids(idx)
            # query image have too many classes, can not find enough
            # negative support classes.
            if len(self.CLASSES) - len(cat_ids) >= self.num_support_ways - 1:
                break
            else:
                idx = self._rand_another(idx) % self._ori_len
            assert try_time < 100, \
                'Not enough negative support classes for query image,' \
                ' please try a smaller support way.'

        query_class = np.random.choice(cat_ids)
        query_gt_idx = [
            i for i in range(len(cat_ids)) if cat_ids[i] == query_class
        ]
        query_data = \
            self.query_dataset.prepare_train_img(idx, 'query', query_gt_idx)
        query_data['query_class'] = [query_class]

        # sample negative support classes, which not appear in query image
        support_class = [
            i for i in range(len(self.CLASSES)) if i not in cat_ids
        ]
        support_class = np.random.choice(
            support_class,
            min(self.num_support_ways - 1, len(support_class)),
            replace=False)
        support_idxes = self.generate_support(idx, query_class, support_class)
        support_data = [
            self.support_dataset.prepare_train_img(idx, 'support', [gt_idx])
            for (idx, gt_idx) in support_idxes
        ]
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self):
        """Length after repetition."""
        return len(self.query_dataset) * self.repeat_times

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
            list[tuple(int)]: A mini-batch (num_support_ways *
                num_support_shots) of support data (idx, gt_idx).
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
            list[tuple[int]]: Support data (num_support_shots)
                of specific class.
        """
        support_idxes = []
        num_total_shots = len(self.data_infos_by_class[class_id])

        # count number of support instance in query image
        cat_ids = self.support_dataset.get_cat_ids(idx % self._ori_len)
        num_ignore_shots = len([1 for cat_id in cat_ids if cat_id == class_id])

        # set num_sample_shots for each time of sampling
        if num_total_shots - num_ignore_shots < self.num_support_shots:
            # if not have enough support data allow repeated data
            num_sample_shots = num_total_shots
            allow_repeat = True
        else:
            # if have enough support data not allow repeated data
            num_sample_shots = self.num_support_shots
            allow_repeat = False
        while len(support_idxes) < self.num_support_shots:
            selected_gt_idxes = np.random.choice(
                num_total_shots, num_sample_shots, replace=False)

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
                if len(support_idxes) == self.num_support_shots:
                    break
            # update the number of data for next time sample
            num_sample_shots = min(self.num_support_shots - len(support_idxes),
                                   num_sample_shots)
        return support_idxes

    def save_data_infos(self, output_path):
        """Save data_infos into json."""
        self.query_dataset.save_data_infos(output_path)
        # for query aware datasets support and query set use same data
        paths = output_path.split('.')
        self.support_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['support_shot', paths[-1]]))

    def get_support_data_infos(self):
        """Return data_infos of support dataset."""
        return self.support_dataset.data_infos


@DATASETS.register_module()
class NwayKshotDataset(object):
    """A dataset wrapper of NwayKshotDataset.

    Building NwayKshotDataset requires query and support dataset, the behavior
    of NwayKshotDataset is determined by `mode`. When dataset in 'query' mode,
    dataset will return regular image and annotations. While dataset in
    'support' mode, dataset will build batch indexes firstly and each batch
    index contain (num_support_ways * num_support_shots) samples. In other
    words, for support mode every call of `__getitem__` will return a batch
    of samples, therefore the outside dataloader should set batch_size to 1.
    The default `mode` of NwayKshotDataset is 'query' and by using convert
    function `convert_query_to_support` the `mode` will be converted into
    'support'.

    Args:
        query_dataset (obj:`FewShotCustomDataset`):
            Query dataset to be wrapped.
        support_dataset (obj:`FewShotCustomDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch.
        num_support_shots (int): Number of support shot for each
            class in mini-batch.
        mutual_support_shot (bool): If True only one annotation will be
            sampled from each image. Default: False.
        num_used_support_shots (int): The total number of support shots
            sampled and used for each class during training. If set to -1,
            all shots in dataset will be used as support shot. Default:-1.
        shuffle_support (bool): If allow generate new batch index for
            each epoch. Default: False.
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset,
                 support_dataset,
                 num_support_ways,
                 num_support_shots,
                 mutual_support_shot=False,
                 num_used_support_shots=None,
                 shuffle_support=False,
                 repeat_times=1):
        self.query_dataset = query_dataset
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        self.CLASSES = self.query_dataset.CLASSES
        # The mode determinate the behavior of fetching data,
        # the default mode is 'query'. To convert the dataset
        # into 'support' dataset, simply call the function
        # convert_query_to_support().
        self.mode = 'query'
        self.num_support_ways = num_support_ways
        self.mutual_support_shot = mutual_support_shot
        self.num_used_support_shots = num_used_support_shots
        self.shuffle_support = shuffle_support
        assert num_support_ways <= len(self.CLASSES), \
            'support way can not larger than the number of classes'
        self.num_support_shots = num_support_shots
        self.batch_index = []
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        self.prepare_support_shots()
        self.repeat_times = repeat_times
        # Disable the groupsampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        self._ori_len = len(self.query_dataset)

    def __getitem__(self, idx):
        if self.mode == 'query':
            idx %= self._ori_len
            # loads one data in query pipeline
            return self.query_dataset.prepare_train_img(idx, 'query')
        elif self.mode == 'support':
            # loads one batch of data in support pipeline
            b_idx = self.batch_index[idx]
            batch_data = [
                self.support_dataset.prepare_train_img(idx, 'support',
                                                       [gt_idx])
                for (idx, gt_idx) in b_idx
            ]
            return batch_data
        else:
            raise ValueError('not valid data type')

    def __len__(self):
        """Length of dataset."""
        if self.mode == 'query':
            return len(self.query_dataset) * self.repeat_times
        elif self.mode == 'support':
            return len(self.batch_index)
        else:
            raise ValueError(f'{self.mode}not a valid mode')

    def prepare_support_shots(self):
        # create lookup table for annotations in same class
        # Support shots are simply loaded in order of data infos
        # until the number met the setting. When `mutual_support_shot`
        # is true, only one annotation will be sampled for each image.
        # TODO: more way to random select support shots
        for idx in range(len(self.support_dataset)):
            labels = self.support_dataset.get_ann_info(idx)['labels']
            for gt_idx, gt in enumerate(labels):
                if self.num_used_support_shots is None or \
                        (len(self.data_infos_by_class[gt]) <
                         self.num_used_support_shots):
                    self.data_infos_by_class[gt].append((idx, gt_idx))
                    if self.mutual_support_shot:
                        break
        # make sure all class index lists have enough
        # instances (length > num_support_shots)
        for i in range(len(self.CLASSES)):
            num_gts = len(self.data_infos_by_class[i])
            if num_gts < self.num_support_shots:
                self.data_infos_by_class[i] = \
                    self.data_infos_by_class[i] * \
                    (self.num_support_shots // num_gts + 1)

    def shuffle_support(self):
        """Generate new batch indexes."""
        if not self.shuffle_support:
            return
        if self.mode == 'query':
            raise ValueError('not support data type')
        self.batch_index = self.generate_index(len(self.batch_index))

    def convert_query_to_support(self, support_dataset_len):
        """Convert query dataset to support dataset.

        Args:
            support_dataset_len (int): Length of pre sample batch indexes.
        """
        self.batch_index = self.generate_index(support_dataset_len)
        self.mode = 'support'
        if hasattr(self, 'flag'):
            self.flag = np.zeros(support_dataset_len, dtype=np.uint8)

    def generate_index(self, dataset_len):
        """Generate batch index [length of datasets * [support way * support shots]].

        Args:
            dataset_len: Length of pre sample batch indexes.

        Returns:
            List[List[(data_idx, gt_idx)]]: Pre sample batch indexes.
        """
        total_index = []
        for _ in range(dataset_len):
            batch_index = []
            selected_classes = np.random.choice(
                len(self.CLASSES), self.num_support_ways, replace=False)
            for cls in selected_classes:
                num_gts = len(self.data_infos_by_class[cls])
                selected_gts_idx = np.random.choice(
                    num_gts, self.num_support_shots, replace=False)
                selected_gts = [
                    self.data_infos_by_class[cls][gt_idx]
                    for gt_idx in selected_gts_idx
                ]
                batch_index.extend(selected_gts)
            total_index.append(batch_index)
        return total_index

    def save_data_infos(self, output_path):
        """Save data infos of query and support data."""
        self.query_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.save_support_data_infos('.'.join(paths[:-1] +
                                              ['support_shot', paths[-1]]))

    def save_support_data_infos(self, support_output_path):
        """Save support data infos."""
        support_data_infos = self.get_support_data_infos()
        meta_info = [{
            'CLASSES': self.CLASSES,
            'img_prefix': self.support_dataset.img_prefix
        }]
        from .utils import NumpyEncoder
        with open(support_output_path, 'w', encoding='utf-8') as f:
            json.dump(
                meta_info + support_data_infos,
                f,
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder)

    def get_support_data_infos(self):
        """Get support data infos from batch index."""
        return [
            self._get_shot_data_info(idx, gt_idx)
            for class_name in self.data_infos_by_class.keys()
            for (idx, gt_idx) in self.data_infos_by_class[class_name]
        ]

    def _get_shot_data_info(self, idx, gt_idx):
        """Get data info by idx and gt idx."""
        data_info = copy.deepcopy(self.support_dataset.data_infos[idx])
        data_info['ann']['labels'] = \
            data_info['ann']['labels'][gt_idx:gt_idx+1]
        data_info['ann']['bboxes'] = \
            data_info['ann']['bboxes'][gt_idx:gt_idx+1]
        return data_info
