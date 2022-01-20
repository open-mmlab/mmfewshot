# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

import mmcv

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument(
        '--shots',
        type=int,
        nargs='+',
        default=[1, 2, 3, 5, 10],
        help='List of shots to generate.')
    parser.add_argument(
        '--root',
        type=str,
        default='./data/VOCdevkit',
        help='Path to dataset.')
    parser.add_argument(
        '--out',
        type=str,
        default='./data/few_shot_ann/voc',
        help='Path to save generated annotations.')
    parser.add_argument(
        '--allow-difficult',
        action='store_true',
        help='Whether to sample difficult instances')
    args = parser.parse_args()
    return args


def main(args):
    file_ids = []
    for year in [2007, 2012]:
        data_file = os.path.join(args.root,
                                 f'VOC{year}/ImageSets/Main/trainval.txt')
        file_ids.extend(mmcv.list_from_file(data_file))

    ann_file_per_class = {c: [] for c in VOC_CLASSES}
    for file_id in file_ids:
        year = '2012' if '_' in file_id else '2007'
        dir_name = os.path.join(args.root, 'VOC{}'.format(year))
        ann_file = os.path.join(dir_name, 'Annotations', file_id + '.xml')
        tree = ET.parse(ann_file)
        classes = []
        for obj in tree.findall('object'):
            classes.append(obj.find('name').text)
        for cls in set(classes):
            ann_file_per_class[cls].append(ann_file)

    result = {cls: {} for cls in VOC_CLASSES}
    random.seed(args.seed)
    for cls in ann_file_per_class.keys():
        sampled_ann_files = []
        sampled_images = []
        for j, shot in enumerate(args.shots):
            diff_shot = args.shots[j] - args.shots[j - 1] if j != 0 else shot
            num_objs = 0
            while num_objs < diff_shot:
                random_ann_file = random.choice(ann_file_per_class[cls])
                if random_ann_file not in sampled_ann_files:
                    sampled_ann_files.append(random_ann_file)
                    tree = ET.parse(random_ann_file)
                    file = tree.find('filename').text
                    year = tree.find('folder').text
                    sampled_images.append(f'{year}/JPEGImages/{file}')
                    for obj in tree.findall('object'):
                        if obj.find('name').text == cls:
                            if args.allow_difficult:
                                num_objs += 1
                            else:
                                difficult = obj.find('difficult')
                                difficult = 0 if difficult is None else int(
                                    difficult.text)
                                if difficult:
                                    continue
                                else:
                                    num_objs += 1
            result[cls][shot] = copy.deepcopy(sampled_ann_files)
    save_paths = {}
    for shot in args.shots:
        if args.allow_difficult:
            save_paths[shot] = os.path.join(
                args.out, f'seed{args.seed}_{shot}shot_with_difficult')
        else:
            save_paths[shot] = os.path.join(args.out,
                                            f'seed{args.seed}_{shot}shot')
        os.makedirs(save_paths[shot], exist_ok=True)
    for cls in result.keys():
        for shot in result[cls].keys():
            filename = 'box_{}shot_{}_train.txt'.format(shot, cls)
            with open(os.path.join(save_paths[shot], filename), 'w') as fp:
                fp.write('\n'.join(result[cls][shot]) + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
