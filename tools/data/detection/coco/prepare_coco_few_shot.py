# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import random

ID2CLASS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}
CLASS2ID = {v: k for k, v in ID2CLASS.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument(
        '--shots',
        type=int,
        nargs='+',
        default=[10, 30],
        help='List of shots to generate.')
    parser.add_argument(
        '--root', type=str, default='./data/coco', help='Path to dataset.')
    parser.add_argument(
        '--out',
        type=str,
        default='./data/few_shot_ann/coco',
        help='Path to save generated annotations.')
    args = parser.parse_args()
    return args


def main(args):
    data = json.load(open(os.path.join(args.root, 'annotations/train.json')))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1:
            continue
        anno[a['category_id']].append(a)

    random.seed(args.seed)
    for cls in ID2CLASS.keys():
        img_ids = {}
        for a in anno[cls]:
            if a['image_id'] in img_ids:
                img_ids[a['image_id']].append(a)
            else:
                img_ids[a['image_id']] = [a]

        sample_shots = []
        sample_imgs = []
        for shot in args.shots:
            while True:
                imgs = random.sample(list(img_ids.keys()), shot)
                for img in imgs:
                    skip = False
                    for s in sample_shots:
                        if img == s['image_id']:
                            skip = True
                            break
                    if skip:
                        continue
                    if len(img_ids[img]) + len(sample_shots) > shot:
                        continue
                    sample_shots.extend(img_ids[img])
                    sample_imgs.append(id2img[img])
                    if len(sample_shots) == shot:
                        break
                if len(sample_shots) == shot:
                    break
            new_data = {
                'info': data['info'],
                'licenses': data['licenses'],
                'images': sample_imgs,
                'annotations': sample_shots,
            }
            save_dir = os.path.join(args.out, f'seed{args.seed}_{shot}shot')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, f'full_box_{shot}shot_{ID2CLASS[cls]}_trainval.json')
            new_data['categories'] = new_all_cats
            with open(save_path, 'w') as f:
                json.dump(new_data, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
