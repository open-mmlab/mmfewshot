# Meta R-CNN: Towards General Solver for Instance-level Low-shot Learning <a href="https://arxiv.org/pdf/1909.13032.pdf"> (ICCV'2019)</a>

## Abstract

<!-- [ABSTRACT] -->

Resembling the rapid learning capability of human, low-shot learning empowers vision systems to understand new concepts by training with few samples.
Leading approaches derived from meta-learning on images with a single visual object.
Obfuscated by a complex background and multiple objects in one image, they are hard to promote the research of low-shot object detection/segmentation.
In this work, we present a flexible and general methodology to achieve these tasks.
Our work extends Faster /Mask R-CNN by proposing meta-learning over RoI (Region-of-Interest) features instead of a full image feature.
This simple spirit disentangles multi-object information merged with the background, without bells and whistles, enabling Faster / Mask R-CNN turn into a meta-learner to achieve the tasks.
Specifically, we introduce a Predictor-head Remodeling Network (PRN) that shares its main backbone with Faster / Mask R-CNN.
PRN receives images containing low-shot objects with their bounding boxes or masks to infer their class attentive vectors.
The vectors take channel-wise soft-attention on RoI features, remodeling those R-CNN predictor heads to detect or segment the objects consistent with the classes these vectors represent.
In our experiments, Meta R-CNN yields the new state of the art in low-shot object detection and improves low-shot object segmentation by Mask R-CNN.
Code: https://yanxp.github.io/metarcnn.html.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142843770-6390a0b2-f40a-4731-ad4d-b6ab4c8268b8.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{yan2019meta,
    title={Meta r-cnn: Towards general solver for instance-level low-shot learning},
    author={Yan, Xiaopeng and Chen, Ziliang and Xu, Anni and Wang, Xiaoxi and Liang, Xiaodan and Lin, Liang},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    year={2019}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.

## How to reproduce Meta RCNN

Following the original implementation, it consists of 2 steps:

- **Step1: Base training**

  - use all the images and annotations of base classes to train a base model.

- **Step2: Few shot fine-tuning**:

  - use the base model from step1 as model initialization and further fine tune the model with few shot datasets.

### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py 8

# step2: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8
```

**Note**:

- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.

## Results on VOC dataset

**Note**:

- The official implementation use batch size 1x4 for training, while we use batch size 8x4.
- For few shot fine-tuning we only fine tune the bbox head and the iterations or training strategy may not be the
  optimal in 8gpu setting.
- Base training use 200 support base instances each class for testing.
- The performance of the base training and few shot setting can be unstable, even using the same random seed.
  To reproduce the reported few shot results, it is highly recommended using the released model for few shot fine-tuning.
- The difficult samples will be used in base training query set, but not be used in support set and few shot setting.

### Base Training

|                                                 Arch                                                  | Split | Base AP50 |                                                                             ckpt                                                                             |                                                                   log                                                                   |
| :---------------------------------------------------------------------------------------------------: | :---: | :-------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py) |   1   |   72.8    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training_20211101_234042-7184a596.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.log.json) |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_base-training.py) |   2   |   73.3    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_base-training_20211101_004034-03616bec.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_base-training.log.json) |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_base-training.py) |   3   |   74.2    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_base-training_20211101_040111-24a50a91.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_base-training.log.json) |

### Few Shot Fine-tuning

|                                                    Arch                                                    | Split | Shot | Base AP50 | Novel AP50 |                                                                               ckpt                                                                                |                                                                     log                                                                      |
| :--------------------------------------------------------------------------------------------------------: | :---: | :--: | :-------: | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py)  |   1   |  1   |   58.8    |    40.2    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning_20211111_173217-b872c72a.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py)  |   1   |  2   |   67.7    |    49.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_2shot-fine-tuning_20211111_173941-75b01b1d.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_2shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py)  |   1   |  3   |   69.0    |    54.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_3shot-fine-tuning_20211111_175026-6b556b8c.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_3shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py)  |   1   |  5   |   70.8    |    55.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning_20211111_180727-d9194139.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py) |   1   |  10  |   71.7    |    56.3    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning_20211111_182413-f3db91b6.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning.log.json) |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_1shot-fine-tuning.py)  |   2   |  1   |   61.0    |    27.3    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_1shot-fine-tuning_20211111_184455-c0319926.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_1shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_2shot-fine-tuning.py)  |   2   |  2   |   69.5    |    34.8    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_2shot-fine-tuning_20211111_185215-c5807bb2.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_2shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_3shot-fine-tuning.py)  |   2   |  3   |   71.0    |    39.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_3shot-fine-tuning_20211111_190314-add8dbf5.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_3shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_5shot-fine-tuning.py)  |   2   |  5   |   71.7    |    36.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_5shot-fine-tuning_20211111_192028-61dcc52f.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_5shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_10shot-fine-tuning.py) |   2   |  10  |   72.6    |    40.1    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_10shot-fine-tuning_20211111_193726-2bc2e6dc.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_10shot-fine-tuning.log.json) |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_1shot-fine-tuning.py)  |   3   |  1   |   63.0    |    32.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_1shot-fine-tuning_20211111_195827-63728ee6.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_1shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_2shot-fine-tuning.py)  |   3   |  2   |   70.1    |    37.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_2shot-fine-tuning_20211111_200558-4ef3a000.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_2shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_3shot-fine-tuning.py)  |   3   |  3   |   71.3    |    42.5    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_3shot-fine-tuning_20211111_201709-eb05339e.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_3shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_5shot-fine-tuning.py)  |   3   |  5   |   72.3    |    49.6    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_5shot-fine-tuning_20211111_203427-54bdf978.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_5shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_10shot-fine-tuning.py) |   3   |  10  |   73.2    |    49.1    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_10shot-fine-tuning_20211111_205129-6d94e3b4.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_10shot-fine-tuning.log.json) |

## Results on COCO dataset

**Note**:

- The official implementation use batch size 1x4 for training, while we use batch size 8x4.
- For few shot fine-tuning we only fine tune the bbox head and the iterations or training strategy may not be the
  optimal in 8gpu setting.
- Base training use 200 support base instances each class for testing.
- The performance of the base training and few shot setting can be unstable, even using the same random seed.
  To reproduce the reported few shot results, it is highly recommended using the released model for few shot fine-tuning.

### Base Training

|                                          Arch                                           | Base mAP |                                                                      ckpt                                                                       |                                                            log                                                             |
| :-------------------------------------------------------------------------------------: | :------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------: |
| [r50 c4](/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.py) |   27.8   | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training_20211102_213915-65a22539.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.log.json) |

Few Shot Finetuning

|                                             Arch                                             | Shot | Base mAP | Novel AP50 |                                                                         ckpt                                                                         |                                                               log                                                               |
| :------------------------------------------------------------------------------------------: | :--: | :------: | :--------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
| [r50 c4](/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.py) |  10  |   25.1   |    9.4     | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning_20211112_090638-e703f762.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_30shot-fine-tuning.py) |  30  |   26.9   |    11.5    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_30shot-fine-tuning_20211112_110452-50d791dd.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_30shot-fine-tuning.log.json) |
