# Few Shot Classification Demo

## 1 shot classification Demo (Metric Based Model)

We provide a demo script to test a single query image, given directory of 1shot support images.
The file names of support images will be used as class names.

```shell
python demo/demo_metric_classifier_1shot_inference.py \
    ${IMG_ROOT} ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    [--device ${GPU_ID or CPU}] \
    [--support-images-dir ${DIR_ROOT}]
```

Examples:

```shell
python demo/demo_metric_classifier_1shot_inference.py \
    demo/demo_classification_images/query_images/Least_Auklet.jpg \
    configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py \
    https://download.openmmlab.com/mmfewshot/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot_20211120_101211-9ab530c3.pth \
    --support-images-dir demo/demo_classification_images/support_images
```

To run demos on CPU:

```shell
python demo/demo_metric_classifier_1shot_inference.py \
    demo/demo_classification_images/query_images/Least_Auklet.jpg \
    configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py \
    https://download.openmmlab.com/mmfewshot/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot_20211120_101211-9ab530c3.pth \
    --support-images-dir demo/demo_classification_images/support_images \
    --device=cpu
```

# Few Shot Detection Demo

## Attention RPN inference with support instances Demo

We provide a demo script to test a single query image, given directory of support instance images.
The file names of support images will be used as class names.
The shape of image will be used as the bbox of instance, i.e [0, 0, width, height].

```shell
python demo/demo_attention_rpn_detector_inference.py \
    ${IMG_ROOT} ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${BBOX_SCORE_THR} --support-images-dir ${DIR_ROOT}]
```

Examples:

```shell
python demo/demo_attention_rpn_detector_inference.py \
    demo/demo_detection_images/query_images/demo_query.jpg \
    configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py \
    https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training_20211102_003348-da28cdfd.pth \
    --support-images-dir demo/demo_detection_images/support_images
```
