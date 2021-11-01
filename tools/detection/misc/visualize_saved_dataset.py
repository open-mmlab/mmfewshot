# Copyright (c) OpenMMLab. All rights reserved.
"""Visualized instances of saved dataset.

Example:
    python tools/detection/misc/visualize_saved_dataset.py \
        --src ./work_dirs/xx_saved_data.json
        --dir ./vis_images
"""
import argparse
import json
import os

import mmcv
import numpy as np
from terminaltables import AsciiTable

try:
    import cv2
except ():
    raise ImportError('please install cv2 mutually')


class Visualizer:
    """Visualize instances of saved dataset.

    Args:
        src (str): Path to saved dataset.
        out_dir (str): Saving directory for output image. Default: ''.
        classes (list[str]): Classes of saved dataset. Default: None.
        img_prefix (str): Prefix for images path. Default: ''.
    """

    def __init__(self, src, out_dir='', classes=None, img_prefix=''):

        if classes is None:
            classes = []
        self.CLASSES = classes
        self.img_prefix = img_prefix
        self.ann_file = src
        self.out_dir = out_dir
        self.data_infos = self.load_annotations_saved(src)
        mmcv.mkdir_or_exist(os.path.abspath(out_dir))
        self.color_map = np.random.randint(0, 255,
                                           (len(self.CLASSES), 3)).tolist()

    def __repr__(self):
        """Print the number of instance number."""
        result = (f'\n dataset statistics'
                  f'with number of images {len(self.data_infos)}, '
                  f'and instance counts: \n')
        if len(self.CLASSES) == 0:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for data_info in self.data_infos:
            label = data_info['ann']['labels']
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

    def visualize(self,
                  save_name='',
                  num_rows=20,
                  num_cols=10,
                  instance_size=256):
        """Visualizing cropped instances in grid layout.

        Instances of same classes will be placed in same row, If the number
        of total classes is larger than `num_rows`, then it will output
        multiple images. If the number of instances is larger than `num_cols`,
        then the exceeded instances will not be visualized.

        Args:
            save_name (str): Name of output image. Default: ''.
            num_rows (int): Number of rows (classes) in single output image.
                Default: 20.
            num_cols (int): Number of column (instances of each class) in
                single output image. Default: 10.
            instance_size (int): Size of cropped instance. Default: 256.
        """
        if save_name == '':
            save_name = os.path.split(self.ann_file)[1].split('.')[0]
        instances = {i: [] for i in range(len(self.CLASSES))}
        for data_info in self.data_infos:
            labels = data_info['ann']['labels']
            bboxes = data_info['ann']['bboxes']
            file_name = data_info['filename']
            image_path = os.path.join(self.img_prefix, file_name)
            img = cv2.imread(image_path)
            for i in range(len(bboxes)):
                bbox = list(map(int, bboxes[i]))
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              self.color_map[labels[i]], 2)
            for i in range(len(labels)):
                croped_instance = self.crop_instance(img, bboxes[i],
                                                     instance_size)
                instances[int(labels[i])].append(croped_instance)
        start, end = 0, 0
        classes = list(instances.keys())
        while end < len(classes):
            if start + num_rows < len(classes):
                end = start + num_rows
            else:
                end = len(classes)
            instances_ = {i: instances[i] for i in classes[start:end]}
            image_name = f'{save_name}_class_{start}_to_{end}.jpg'
            img_out = self.concat_images(instances_, num_cols, instance_size)
            image_path = os.path.join(self.out_dir, image_name)
            cv2.imwrite(image_path, img_out)
            print(f'image {image_name} is saved to {self.out_dir}')
            start = end

    @staticmethod
    def crop_instance(img, bbox, instance_size=256):
        """Crop and resize instance.

        Args:
            img (numpy.ndarray): Image to be cropped.
            bbox (list[int]): BBox of instance.
            instance_size (int): Resize cropped instance to `instance_size`.
                Default: 256.

        Returns:
            numpy.ndarray: cropped instance.
        """
        bbox = list(map(int, bbox))
        h, w = img.shape[0], img.shape[1]
        b_h = bbox[3] - bbox[1]
        b_w = bbox[2] - bbox[0]
        pad_y = (instance_size -
                 b_h) // 2 if b_h < instance_size else b_h * 0.1
        pad_x = (instance_size -
                 b_w) // 2 if b_w < instance_size else b_w * 0.1
        if b_h > b_w:
            pad_y = int(pad_y)
            target_size = pad_y * 2 + b_h
            pad_x = int((target_size - b_w) // 2)
            region = np.zeros((target_size + 4, target_size + 4, 3), np.uint8)
        else:
            pad_x = int(pad_x)
            target_size = pad_x * 2 + b_w
            pad_y = int((target_size - b_h) // 2)
            region = np.zeros((target_size + 4, target_size + 4, 3), np.uint8)
        y0 = bbox[1] - pad_y if bbox[1] - pad_y > 0 else 0
        y1 = bbox[3] + pad_y if bbox[3] + pad_y < h else h
        x0 = bbox[0] - pad_x if bbox[0] - pad_x > 0 else 0
        x1 = bbox[2] + pad_x if bbox[2] + pad_x < w else w
        t_y0 = (target_size - (y1 - y0)) // 2
        t_y1 = t_y0 + (y1 - y0)
        t_x0 = (target_size - (x1 - x0)) // 2
        t_x1 = t_x0 + (x1 - x0)
        region[t_y0:t_y1, t_x0:t_x1] = img[y0:y1, x0:x1]
        region = cv2.resize(region, (instance_size, instance_size))

        return region

    def concat_images(self, cropped_images, num_cols, offset):
        """Concat cropped cropped_images in grid layout.

        Args:
            cropped_images (dict): Images of cropped instance.
            num_cols (int): Number of columns of grid layout.
            offset (int): Size of each cropped instance.

        Returns:
            numpy.ndarray: visualized image.
        """
        num_rows = len(cropped_images.keys())
        num_cols = num_cols + 1
        canvas = np.zeros((num_rows * offset, num_cols * offset, 3), np.uint8)
        for row, label in enumerate(cropped_images.keys()):
            label_image = np.full((offset, offset, 3), 255, dtype=np.uint8)
            class_name = self.CLASSES[label]
            label_image = cv2.putText(label_image, class_name,
                                      (10, offset // 2),
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                                      2, cv2.LINE_AA)
            canvas[offset * row:offset * (row + 1), 0:offset, :] = label_image
            for col in range(1, num_cols):
                if col > len(cropped_images[label]):
                    img = np.zeros((offset, offset, 3), np.uint8)
                else:
                    img = cropped_images[label][col - 1]
                canvas[offset * row:offset * (row + 1),
                       offset * col:offset * (col + 1), :] = img
        return canvas

    def load_annotations_saved(self, ann_file):
        """Load data_infos from saved json."""
        with open(ann_file) as f:
            data_infos = json.load(f)
        meta_index = None
        for i, data_info in enumerate(data_infos):
            if 'CLASSES' in data_info.keys():
                self.CLASSES = data_info['CLASSES']
                if 'img_prefix' in data_info.keys():
                    self.img_prefix = data_info['img_prefix']
                meta_index = i
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
        assert len(self.CLASSES) > 0, 'missing CLASSES for saved dataset json.'
        if meta_index is not None:
            data_infos.pop(meta_index)
        return data_infos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a saved FewShot Dataset')
    parser.add_argument(
        '--src', type=str, help='saved few shot dataset file path')
    parser.add_argument(
        '--dir', type=str, help='output dir to save visualize images')
    parser.add_argument(
        '--save-name',
        default='',
        type=str,
        help='saved name of visualize images')
    parser.add_argument(
        '--row',
        default=20,
        type=int,
        help='number of classes to show in one image')
    parser.add_argument(
        '--col',
        default=10,
        type=int,
        help='number of instance to show for each class')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    visualizer = Visualizer(args.src, args.dir)
    print(visualizer)
    visualizer.visualize(args.save_name, num_rows=args.row, num_cols=args.col)
