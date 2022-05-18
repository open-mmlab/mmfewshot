# Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild <a href="https://arxiv.org/abs/2007.12107"> (ECCV'2020)</a>

## Abstract

<!-- [ABSTRACT] -->

Detecting objects and estimating their viewpoint in images are key tasks of 3D scene understanding.
Recent approaches have achieved excellent results on very large benchmarks for object detection and view-point estimation.
However, performances are still lagging behind for novel object categories with few samples.
In this paper, we tackle the problems of few-shot object detection and few-shot viewpoint estimation.
We propose a meta-learning framework that can be applied to both tasks, possibly including 3D data.
Our models improve the results on objects of novel classes by leveraging on rich feature information originating from base classes with many samples. A simple joint
feature embedding module is proposed to make the most of this feature sharing.
Despite its simplicity, our method outperforms state-of-the-art methods by a large margin on a range of datasets, including
PASCAL VOC and MS COCO for few-shot object detection, and Pascal3D+ and ObjectNet3D for few-shot viewpoint estimation.
And for the first time, we tackle the combination of both few-shot tasks, on ObjectNet3D, showing promising results.
Our code and data are available at http://imagine.enpc.fr/~xiaoy/FSDetView/.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142845154-a50d8902-1b7a-4c7e-9b36-0848ff080187.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2020fsdetview,
    title={Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild},
    author={Yang Xiao and Renaud Marlet},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2020}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.

## How to reproduce FSDetView

Following the original implementation, it consists of 2 steps:

- **Step1: Base training**

  - use all the images and annotations of base classes to train a base model.

- **Step2: Few shot fine-tuning**:

  - use the base model from step1 as model initialization and further fine tune the model with few shot datasets.

### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.py 8

# step2: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8
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
| [r101 c4](/configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.py) |   1   |   73.7    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training_20211101_072143-6d1fd09d.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.log.json) |
| [r101 c4](/configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_base-training.py) |   2   |   74.6    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_base-training_20211101_104321-6890f5da.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_base-training.log.json) |
| [r101 c4](/configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_base-training.py) |   3   |   73.2    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_base-training_20211101_140448-c831f0cf.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_base-training.log.json) |

### Few Shot Fine-tuning

|                                                    Arch                                                    | Split | Shot | Base AP50 | Novel AP50 |                                                                               ckpt                                                                                |                                                                     log                                                                      |
| :--------------------------------------------------------------------------------------------------------: | :---: | :--: | :-------: | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
| [r101 c4](/configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py)  |   1   |  1   |   61.1    |    35.5    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning_20211111_174458-7f003e09.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py)  |   1   |  2   |   67.9    |    49.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_2shot-fine-tuning_20211111_175533-6a218bf8.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_2shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py)  |   1   |  3   |   68.1    |    54.6    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_3shot-fine-tuning_20211111_181021-7fed633f.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_3shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py)  |   1   |  5   |   69.9    |    60.5    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_5shot-fine-tuning_20211111_183331-2bad7372.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_5shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py) |   1   |  10  |   71.7    |    61.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning_20211111_185540-3717717b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning.log.json) |
| [r101 c4](/configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_1shot-fine-tuning.py)  |   2   |  1   |   65.0    |    27.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_1shot-fine-tuning_20211111_192244-0fb62181.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_1shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_2shot-fine-tuning.py)  |   2   |  2   |   69.6    |    36.6    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_2shot-fine-tuning_20211111_193302-77a3e0ed.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_2shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_3shot-fine-tuning.py)  |   2   |  3   |   70.9    |    41.4    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_3shot-fine-tuning_20211111_194805-a9746764.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_3shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_5shot-fine-tuning.py)  |   2   |  5   |   71.3    |    43.2    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_5shot-fine-tuning_20211111_201121-627d8bab.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_5shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_10shot-fine-tuning.py) |   2   |  10  |   72.4    |    47.8    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_10shot-fine-tuning_20211111_203317-1d8371fa.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_10shot-fine-tuning.log.json) |
| [r101 c4](/configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_1shot-fine-tuning.py)  |   3   |  1   |   61.4    |    37.3    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_1shot-fine-tuning_20211111_210030-71f567aa.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_1shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_2shot-fine-tuning.py)  |   3   |  2   |   69.3    |    44.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_2shot-fine-tuning_20211111_211043-903328e6.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_2shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_3shot-fine-tuning.py)  |   3   |  3   |   70.5    |    47.5    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_3shot-fine-tuning_20211111_212549-c2a73819.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_3shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_5shot-fine-tuning.py)  |   3   |  5   |   72.2    |    52.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_5shot-fine-tuning_20211111_214912-2650edee.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_5shot-fine-tuning.log.json)  |
| [r101 c4](/configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_10shot-fine-tuning.py) |   3   |  10  |   73.1    |    52.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_10shot-fine-tuning_20211111_221125-7f2f0ddb.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_10shot-fine-tuning.log.json) |

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
| [r50 c4](/configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_base-training.py) |   21.3   | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_base-training_20211113_011123-02c00ddc.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_base-training.log.json) |

### Few Shot Fine-tuning

|                                             Arch                                             | Shot | Base mAP | Novel mAP |                                                                         ckpt                                                                         |                                                               log                                                               |
| :------------------------------------------------------------------------------------------: | :--: | :------: | :-------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
| [r50 c4](/configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning.py) |  10  |   21.1   |    9.1    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning_20211114_002725-a3c97004.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_30shot-fine-tuning.py) |  30  |   23.5   |   12.4    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_30shot-fine-tuning_20211114_022948-8e0e6378.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_30shot-fine-tuning.log.json) |
