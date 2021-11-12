# Copyright (c) OpenMMLab. All rights reserved.
"""Inference One Shot Classifier with support shots.

Example:
    python demo/demo_metric_classifier_1shot_inference.py \
        demo/demo_classification_images/query_images/Least_Auklet.jpg
        configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py
        ./work_dirs/proto-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth
"""  # nowq

import os
from argparse import ArgumentParser

from mmfewshot.classification.apis import (inference_classifier,
                                           init_classifier,
                                           process_support_images,
                                           show_result_pyplot)


def main():
    parser = ArgumentParser('N way 1 shot inference.')
    parser.add_argument('image', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--support-images-dir',
        default='demo/demo_classification_images/support_images',
        help='path to support images directory, each image will use'
        'file name as class')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_classifier(args.config, args.checkpoint, device=args.device)
    # prepare support set, each support class only contains one shot
    files = os.listdir(args.support_images_dir)
    support_images = [
        os.path.join(args.support_images_dir, file) for file in files
    ]
    support_labels = [file.split('.')[0] for file in files]
    process_support_images(model, support_images, support_labels)
    # test a single image
    result = inference_classifier(model, args.image)
    # show the results
    show_result_pyplot(args.image, result)


if __name__ == '__main__':
    main()
