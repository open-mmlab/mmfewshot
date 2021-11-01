# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Tuple

import mmcv
import numpy as np
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import (Normalize, Pad, RandomCrop, RandomFlip,
                                      Resize)


# TODO: Simplify pipelines by decoupling operation.
@PIPELINES.register_module()
class CropResizeInstance:
    """Crop and resize instance according to bbox form image.

    Args:
        num_context_pixels (int): Padding pixel around instance. Default: 16.
        target_size (tuple[int, int]): Resize cropped instance to target size.
            Default: (320, 320).
    """

    def __init__(
        self,
        num_context_pixels: int = 16,
        target_size: Tuple[int] = (320, 320)
    ) -> None:
        assert isinstance(num_context_pixels, int)
        assert len(target_size) == 2, 'target_size'
        self.num_context_pixels = num_context_pixels
        self.target_size = target_size

    def __call__(self, results: Dict) -> Dict:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Cropped and resized instance results.
        """
        img = results['img']
        gt_bbox = results['gt_bboxes']
        img_h, img_w = img.shape[:2]  # h, w
        x1, y1, x2, y2 = list(map(int, gt_bbox.tolist()[0]))

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        t_x1, t_y1, t_x2, t_y2 = 0, 0, bbox_w, bbox_h

        if bbox_w >= bbox_h:
            crop_x1 = x1 - self.num_context_pixels
            crop_x2 = x2 + self.num_context_pixels
            # t_x1 and t_x2 will change when crop context or overflow
            t_x1 = t_x1 + self.num_context_pixels
            t_x2 = t_x1 + bbox_w
            if crop_x1 < 0:
                t_x1 = t_x1 + crop_x1
                t_x2 = t_x1 + bbox_w
                crop_x1 = 0
            if crop_x2 > img_w:
                crop_x2 = img_w

            short_size = bbox_h
            long_size = crop_x2 - crop_x1
            y_center = int((y2 + y1) / 2)  # math.ceil((y2 + y1) / 2)
            crop_y1 = int(
                y_center -
                (long_size / 2))  # int(y_center - math.ceil(long_size / 2))
            crop_y2 = int(
                y_center +
                (long_size / 2))  # int(y_center + math.floor(long_size / 2))

            # t_y1 and t_y2 will change when crop context or overflow
            t_y1 = t_y1 + math.ceil((long_size - short_size) / 2)
            t_y2 = t_y1 + bbox_h

            if crop_y1 < 0:
                t_y1 = t_y1 + crop_y1
                t_y2 = t_y1 + bbox_h
                crop_y1 = 0
            if crop_y2 > img_h:
                crop_y2 = img_h

            crop_short_size = crop_y2 - crop_y1
            crop_long_size = crop_x2 - crop_x1

            square = np.zeros((crop_long_size, crop_long_size, 3),
                              dtype=np.uint8)
            delta = int(
                (crop_long_size - crop_short_size) /
                2)  # int(math.ceil((crop_long_size - crop_short_size) / 2))
            square_y1 = delta
            square_y2 = delta + crop_short_size

            t_y1 = t_y1 + delta
            t_y2 = t_y2 + delta

            crop_box = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            square[square_y1:square_y2, :, :] = crop_box
        else:
            crop_y1 = y1 - self.num_context_pixels
            crop_y2 = y2 + self.num_context_pixels

            # t_y1 and t_y2 will change when crop context or overflow
            t_y1 = t_y1 + self.num_context_pixels
            t_y2 = t_y1 + bbox_h
            if crop_y1 < 0:
                t_y1 = t_y1 + crop_y1
                t_y2 = t_y1 + bbox_h
                crop_y1 = 0
            if crop_y2 > img_h:
                crop_y2 = img_h

            short_size = bbox_w
            long_size = crop_y2 - crop_y1
            x_center = int((x2 + x1) / 2)  # math.ceil((x2 + x1) / 2)
            crop_x1 = int(
                x_center -
                (long_size / 2))  # int(x_center - math.ceil(long_size / 2))
            crop_x2 = int(
                x_center +
                (long_size / 2))  # int(x_center + math.floor(long_size / 2))

            # t_x1 and t_x2 will change when crop context or overflow
            t_x1 = t_x1 + math.ceil((long_size - short_size) / 2)
            t_x2 = t_x1 + bbox_w
            if crop_x1 < 0:
                t_x1 = t_x1 + crop_x1
                t_x2 = t_x1 + bbox_w
                crop_x1 = 0
            if crop_x2 > img_w:
                crop_x2 = img_w

            crop_short_size = crop_x2 - crop_x1
            crop_long_size = crop_y2 - crop_y1
            square = np.zeros((crop_long_size, crop_long_size, 3),
                              dtype=np.uint8)
            delta = int(
                (crop_long_size - crop_short_size) /
                2)  # int(math.ceil((crop_long_size - crop_short_size) / 2))
            square_x1 = delta
            square_x2 = delta + crop_short_size

            t_x1 = t_x1 + delta
            t_x2 = t_x2 + delta
            crop_box = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            square[:, square_x1:square_x2, :] = crop_box

        square = square.astype(np.float32, copy=False)
        square, square_scale = mmcv.imrescale(
            square, self.target_size, return_scale=True, backend='cv2')

        square = square.astype(np.uint8)

        t_x1 = int(t_x1 * square_scale)
        t_y1 = int(t_y1 * square_scale)
        t_x2 = int(t_x2 * square_scale)
        t_y2 = int(t_y2 * square_scale)
        results['img'] = square
        results['img_shape'] = square.shape
        results['gt_bboxes'] = np.array([[t_x1, t_y1, t_x2,
                                          t_y2]]).astype(np.float32)

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(num_context_pixels={self.num_context_pixels},' \
               f' target_size={self.target_size})'


@PIPELINES.register_module()
class GenerateMask:
    """Resize support image and generate a mask.

    Args:
        target_size (tuple[int, int]): Crop and resize to target size.
            Default: (224, 224).
    """

    def __init__(self, target_size: Tuple[int] = (224, 224)) -> None:
        self.target_size = target_size

    def _resize_bboxes(self, results: Dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            results[key] = bboxes

    def _resize_img(self, results: Dict) -> None:
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results[key],
                self.target_size,
                return_scale=True,
                backend='cv2')
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor

    def _generate_mask(self, results: Dict) -> Dict:
        mask = np.zeros(self.target_size, dtype=np.float32)
        gt_bboxes = results['gt_bboxes'][0]
        mask[int(gt_bboxes[1]):int(gt_bboxes[3]),
             int(gt_bboxes[0]):int(gt_bboxes[2])] = 1
        results['img'] = np.concatenate(
            [results['img'], np.expand_dims(mask, axis=2)], axis=2)
        results['img_shape'] = results['img'].shape
        return results

    def __call__(self, results: Dict) -> Dict:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized images with additional dimension of bbox mask.
        """
        self._resize_img(results)
        self._resize_bboxes(results)
        self._generate_mask(results)

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(num_context_pixels={self.num_context_pixels},' \
               f' target_size={self.target_size})'


@PIPELINES.register_module()
class CropInstance:
    """Crop single instance according to bboxe to form an image.

    Args:
        context_ratio (float): Expand the gt_bboxes of instances to
            (1 + context_ratio) times the original longest side.
            Default: 0.
    """

    def __init__(self, context_ratio: float = 0) -> None:
        assert context_ratio > 0
        self.context_ratio = context_ratio

    def __call__(self, results: Dict) -> Dict:
        """Crop instance according to bbox form image, the padding region
        outside the image will be set to zero.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Cropped instance results.
        """
        img = results['img']
        gt_bbox = results['gt_bboxes']
        assert gt_bbox.shape[0] == 1, \
            'CropInstance pipeline do not accept multiple gt_bboxes as input.'
        h, w = img.shape[:2]
        x1, y1, x2, y2 = gt_bbox[0].tolist()
        crop_size = int(max(x2 - x1, y2 - y1) * (1 + self.context_ratio))
        crop_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        old_x1 = int((x1 + x2 - crop_size) / 2)
        old_y1 = int((y1 + y2 - crop_size) / 2)
        x_shift = x1 - old_x1
        y_shift = y1 - old_y1
        new_x1 = 0 if old_x1 >= 0 else 0 - old_x1
        new_y1 = 0 if old_y1 >= 0 else 0 - old_y1
        old_x1 = max(0, old_x1)
        old_y1 = max(0, old_y1)

        old_x2 = min(w, int((x1 + x2 + crop_size) / 2))
        old_y2 = min(h, int((y1 + y2 + crop_size) / 2))
        new_x2 = new_x1 + old_x2 - old_x1
        new_y2 = new_y1 + old_y2 - old_y1
        crop_img[int(new_y1):int(new_y2), int(new_x1):int(new_x2)] = \
            img[int(old_y1):int(old_y2), int(old_x1):int(old_x2)]
        results['gt_bboxes'] = np.array(
            [[x_shift, y_shift, x2 - x1 + x_shift,
              y2 - y1 + y_shift]]).astype(np.float32)
        results['img'] = crop_img
        results['img_shape'] = crop_img.shape
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(context_ratio={self.context_ratio}'


@PIPELINES.register_module()
class ResizeToMultiScale(Resize):
    """Resize images, bounding boxes, masks, semantic segmentation maps to
    multiple scales.

    Args:
        multi_img_scales (list[tuple(int)]): Multiple scales to resize.
    """

    def __init__(self, multi_scales: List[Tuple[int]], *args,
                 **kwargs) -> None:
        super(ResizeToMultiScale, self).__init__(*args, **kwargs)
        assert isinstance(multi_scales, list)
        assert len(multi_scales) > 1
        self.multi_scales = multi_scales

    def __call__(self, results: Dict) -> List[Dict]:
        """Resize images, bounding boxes, masks, semantic segmentation map with
        multiple scales and return a list of results at multiple scales.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            list[dict]: List of resized results, 'img_shape', 'pad_shape',
            'scale_factor', 'keep_ratio' keys are added into each result
            dict.
        """
        results_list = []
        for scale in self.multi_scales:
            results_ = copy.deepcopy(results)
            results_['scale'] = scale
            self._resize_img(results_)
            self._resize_bboxes(results_)
            self._resize_masks(results_)
            self._resize_seg(results_)
            results_list.append(results_)
        return results_list

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(multi_img_scales={self.multi_img_scales}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class MultiImageRandomCrop(RandomCrop):
    """Random crop the image & bboxes & masks for data at multiple scales.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.

    Args:
        multi_crop_sizes (list[tuple(int)]): Crop size of each scales.
        allow_negative_crop (bool): Whether to allow a crop that does
            not contain any bbox area. Default: False.
        bbox_clip_border (bool): Whether clip the objects outside
            the border of the image. Default: True.
    """

    def __init__(self,
                 multi_crop_sizes: List[Tuple[int]],
                 allow_negative_crop: bool = False,
                 bbox_clip_border: bool = True) -> None:
        assert isinstance(multi_crop_sizes, list)
        assert len(multi_crop_sizes) > 1
        self.multi_crop_sizes = multi_crop_sizes
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results_list: List[Dict]) -> List[Dict]:
        """Randomly crop image, bounding boxes, masks, semantic segmentation
        maps of each results in `results_list`.

        Args:
            results_list (list[dict]): List of result dict from loading
                pipeline.

        Returns:
            list[dict]: Randomly cropped `results_list`, 'img_shape' key in
            each result dict is updated with corresponding crop size.
        """
        for results, crop_size in zip(results_list, self.multi_crop_sizes):
            h, w = results['img'].shape[:2]
            crop_size = (min(crop_size[0], h), min(crop_size[1], w))
            self._crop_data(results, crop_size, self.allow_negative_crop)
        return results_list

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(multi_crop_sizes={self.multi_crop_sizes}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class MultiImageRandomFlip(RandomFlip):

    def __call__(self, results_list: List[Dict]) -> List[Dict]:
        """Random Flip image of each results in `results_list`.

        Args:
            results_list (list[dict]): List of result dict from
                loading pipeline.

        Returns:
            list[dict]: List of normalized results, 'img_norm_cfg' key
            is added into each result dict.
        """
        for results in results_list:
            super().__call__(results)
        return results_list


@PIPELINES.register_module()
class MultiImageNormalize(Normalize):

    def __call__(self, results_list: List[Dict]) -> List[Dict]:
        """Normalize image of each results in `results_list`.

        Args:
            results_list (list[dict]): List of result dict from
                loading pipeline.

        Returns:
            list[dict]: List of normalized results, 'img_norm_cfg' key
                is added into each result dict.
        """
        for results in results_list:
            super().__call__(results)
        return results_list


@PIPELINES.register_module()
class MultiImagePad(Pad):

    def __call__(self, results_list: List[Dict]) -> List[Dict]:
        """Pad images, masks, semantic segmentation maps of each results in
        `results_list`.

        Args:
            results_list (list[dict]): List of result dict from
                loading pipeline.

        Returns:
            list[dict]: List of padded results.
        """
        for results in results_list:
            super().__call__(results)
        return results_list
