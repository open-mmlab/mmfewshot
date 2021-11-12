# Copyright (c) OpenMMLab. All rights reserved.
"""Inference Attention RPN Detector with support instances.

Example:
    python demo/demo_attention_rpn_detector_inference.py \
        ./demo/demo_detection_images/query_images/demo_query.jpg
        configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py
        ./work_dirs/attention-rpn_r50_c4_4xb2_coco-base-training/latest.pth
"""  # nowq

import os
from argparse import ArgumentParser

from mmdet.apis import show_result_pyplot

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                      process_support_images)


def parse_args():
    parser = ArgumentParser('attention rpn inference.')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--support-imgs-dir',
        default='demo/demo_detection_images/support_images',
        help='Image file')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # prepare support images, each demo image only contain one instance
    files = os.listdir(args.support_imgs_dir)
    support_imgs = [
        os.path.join(args.support_imgs_dir, file) for file in files
    ]
    classes = [file.split('.')[0] for file in files]
    support_labels = [[file.split('.')[0]] for file in files]
    process_support_images(
        model, support_imgs, support_labels, classes=classes)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
