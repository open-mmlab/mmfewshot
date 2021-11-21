# Copyright (c) OpenMMLab. All rights reserved.
"""Unzip tiered imagenet dataset from pickle file."""

import argparse
import os
import pickle

import mmcv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        default='data/tiered_imagenet',
        help='the directory to tiered imagenet')
    args = parser.parse_args()
    data_prefix = args.dir
    for subset in ['train', 'test', 'val']:
        img_bytes_file = os.path.join(data_prefix, f'{subset}_images_png.pkl')
        os.makedirs((os.path.join(data_prefix, subset)), exist_ok=True)
        print(f'unzipping {subset} file...')
        with open(img_bytes_file, 'rb') as img_bytes:
            img_bytes = pickle.load(img_bytes)
            prog_bar = mmcv.ProgressBar(len(img_bytes))
            for i in range(len(img_bytes)):
                filename = os.path.join(data_prefix, subset,
                                        f'{subset}_image_{i}.byte')
                # write bytes to file
                with open(filename, 'wb') as binary_file:
                    binary_file.write(img_bytes[i])
                prog_bar.update()


if __name__ == '__main__':
    main()
