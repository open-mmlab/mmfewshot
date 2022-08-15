# FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding <a href="https://arxiv.org/abs/2103.05950"> (CVPR'2021)</a>

## Abstract

<!-- [ABSTRACT] -->

Emerging interests have been brought to recognize previously unseen objects given very few training examples, known as few-shot object detection (FSOD).
Recent researches demonstrate that good feature embedding is
the key to reach favorable few-shot learning performance.
We observe object proposals with different Intersection-of-Union (IoU) scores are analogous to the intra-image augmentation
used in contrastive approaches. And we exploit this analogy and incorporate supervised contrastive learning
to achieve more robust objects representations in FSOD.
We present Few-Shot object detection via Contrastive proposals
Encoding (FSCE), a simple yet effective approach to
learning contrastive-aware object proposal encodings that
facilitate the classification of detected objects. We notice
the degradation of average precision (AP) for rare objects
mainly comes from misclassifying novel instances as confusable
classes. And we ease the misclassification issues
by promoting instance level intra-class compactness and
inter-class variance via our contrastive proposal encoding
loss (CPE loss). Our design outperforms current state-ofthe-art works in any shot and all data splits, with up to
+8.8% on standard benchmark PASCAL VOC and +2.7% on challenging COCO benchmark. Code is available at: https://github.com/bsun0802/FSCE.git

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142846618-074a4b8b-c5fa-474d-a0fd-df724c54f72c.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibteN
@inproceedings{sun2021fsce,
    title={FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding},
    author={Sun, Bo and Li, Banghuai and Cai, Shengcai and Yuan, Ye and Zhang, Chi},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
    year={2021}
}
```

**Note**: ALL the reported results use the data split released from [fsce](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo, unless stated otherwise.
Currently, each setting is only evaluated with one fiNed few shot dataset.
Please refer to [here](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.

## How to reproduce FSCE

Following the original implementation, it consists of 3 steps:

- **Step1: Base training**

  - use all the images and annotations of base classes to train a base model.

- **Step2: Reshape the bbox head of base model**:

  - create a new bbox head for all classes fine-tuning (base classes + novel classes)
  - the weights of base class in new bbox head directly use the original one as initialization.
  - the weights of novel class in new bbox head use random initialization.

- **Step3: Few shot fine-tuning**:

  - use the base model from step2 as model initialization and further fine tune the bbox head with few shot datasets.

### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py 8

# step2: reshape the bbox head of base model for few shot fine-tuning
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/fsce_r101_fpn_voc-split1_base-training/latest.pth \
    --method random_init \
    --save-dir work_dirs/fsce_r101_fpn_voc-split1_base-training

# step3: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py 8
```

**Note**:

- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.

## Results on VOC dataset

### Base Training

|                                           arch                                           | contrastive loss | Split | Base AP50 |                                                                 ckpt(step1)                                                                  |                                                                            ckpt(step2)                                                                             |                                                           log                                                           |
| :--------------------------------------------------------------------------------------: | :--------------: | :---: | :-------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
| [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py) |        N         |   1   |   80.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training_20211031_114821-efbd13e9.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training_20211031_114821_random-init-bbox-head-1e681852.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.log.json) |
| [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_base-training.py) |        N         |   2   |   82.0    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training_20211031_114820-d47f8ef9.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training_20211031_114820_random-init-bbox-head-3d4c632c.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training.log.json) |
| [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_base-training.py) |        N         |   3   |   82.1    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training_20211031_114840-fd8a9864.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training_20211031_114840_random-init-bbox-head-9bb8c09b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training.log.json) |

**Note**:

- All the base training configs is the same as [TFA](https://github.com/open-mmlab/mmfewshot/tree/main/configs/detection/fsce/README.md).
  Therefore, the few shot fine-tuning can directly reuse the reshaped
  base model of fsce by creating a symlink or copying the whole checkpoint to the corresponding folder.
  Also, the released base training checkpoint is the same as the TFA, too.
- The performance of the same few shot setting using different base training models can be dramatically unstable
  (AP50 can fluctuate by 5.0 or more), even their mAP on base classes are very close.
- Temporally, the solution to getting a good base model is training the base model with different random seed.
  Also, the random seed used in this code base may not the optimal one, and it is possible to get the higher results by using
  other random seeds.
  However, using the same random seed still can not guarantee the identical result each time, as some nondeterministic CUDA operations.
  We will continue to investigate and improve it.
- To reproduce the reported few shot results, it is highly recommended using the released step2 model for few shot fine-tuning.
- The difficult samples will be used in base training, but not be used in few shot setting.

### Few Shot Fine-tuning

|                                                      arch                                                      | contrastive loss | Split | Shot | Base AP50 | Novel AP50 |                                                                                 ckpt                                                                                 |                                                                       log                                                                       |
| :------------------------------------------------------------------------------------------------------------: | :--------------: | :---: | :--: | :-------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|          [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py)          |        N         |   1   |  1   |   78.4    |    41.2    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning_20211101_145649-fa1f3164.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_2shot-fine-tuning.py)          |        N         |   1   |  2   |   77.8    |    51.1    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_2shot-fine-tuning_20211101_151949-cc763dba.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_2shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning.py)          |        N         |   1   |  3   |   76.1    |    49.3    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning_20211101_174521-2d12c41b.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning.py)          |        N         |   1   |  5   |   75.9    |    59.4    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning_20211101_181628-3e6bb8fe.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning.log.json)          |
|         [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_10shot-fine-tuning.py)          |        N         |   1   |  10  |   76.4    |    62.6    |         [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_10shot-fine-tuning_20211101_185037-b8635ce5.pth)          |         [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_10shot-fine-tuning.log.json)          |
| [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning.py)  |        Y         |   1   |  3   |   75.0    |    48.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning_20211101_154514-59838a14.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning.log.json)  |
| [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning.py)  |        Y         |   1   |  5   |   75.0    |    58.8    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning_20211101_161702-67cc5b36.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning.log.json)  |
| [r101_fpn](/configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_10shot-fine-tuning.py) |        Y         |   1   |  10  |   75.5    |    63.3    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_10shot-fine-tuning_20211101_165137-833012d3.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_10shot-fine-tuning.log.json) |
|          [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_1shot-fine-tuning.py)          |        N         |   2   |  1   |   79.8    |    25.0    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_1shot-fine-tuning_20211101_194330-9aca29bf.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_1shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_2shot-fine-tuning.py)          |        N         |   2   |  2   |   78.0    |    30.6    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_2shot-fine-tuning_20211101_195856-3e4cbf81.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_2shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_3shot-fine-tuning.py)          |        N         |   2   |  3   |   76.4    |    43.4    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_3shot-fine-tuning_20211101_221253-c3cb1bc5.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_3shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_5shot-fine-tuning.py)          |        N         |   2   |  5   |   77.2    |    45.3    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_5shot-fine-tuning_20211101_224701-36a1b478.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_5shot-fine-tuning.log.json)          |
|         [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_10shot-fine-tuning.py)          |        N         |   2   |  10  |   77.5    |    50.4    |         [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_10shot-fine-tuning_20211101_232105-3f91d0cc.pth)          |         [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_10shot-fine-tuning.log.json)          |
| [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_3shot-fine-tuning.py)  |        Y         |   2   |  3   |   76.3    |    43.3    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_3shot-fine-tuning_20211101_201853-665e5ffb.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_3shot-fine-tuning.log.json)  |
| [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_5shot-fine-tuning.py)  |        Y         |   2   |  5   |   76.6    |    45.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_5shot-fine-tuning_20211101_205345-cfedd8c2.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_5shot-fine-tuning.log.json)  |
| [r101_fpn](/configs/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_10shot-fine-tuning.py) |        Y         |   2   |  10  |   76.8    |    50.4    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_10shot-fine-tuning_20211101_212829-afca4e8e.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_10shot-fine-tuning.log.json) |
|          [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_1shot-fine-tuning.py)          |        N         |   3   |  1   |   79.0    |    39.8    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_1shot-fine-tuning_20211101_145152-5ad96c55.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_1shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_2shot-fine-tuning.py)          |        N         |   3   |  2   |   78.4    |    41.5    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_2shot-fine-tuning_20211101_151930-77eb48e7.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_2shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_3shot-fine-tuning.py)          |        N         |   3   |  3   |   76.1    |    47.1    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_3shot-fine-tuning_20211101_180143-0e3f0471.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_3shot-fine-tuning.log.json)          |
|          [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_5shot-fine-tuning.py)          |        N         |   3   |  5   |   77.4    |    54.1    |          [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_5shot-fine-tuning_20211101_183836-b25db64d.pth)          |          [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_5shot-fine-tuning.log.json)          |
|         [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_10shot-fine-tuning.py)          |        N         |   3   |  10  |   77.7    |    57.4    |         [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_10shot-fine-tuning_20211101_192133-f56834f6.pth)          |         [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_10shot-fine-tuning.log.json)          |
| [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_3shot-fine-tuning.py)  |        Y         |   3   |  3   |   75.6    |    48.1    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_3shot-fine-tuning_20211101_154634-4ba95ebb.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_3shot-fine-tuning.log.json)  |
| [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_5shot-fine-tuning.py)  |        Y         |   3   |  5   |   76.2    |    55.7    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_5shot-fine-tuning_20211101_162401-7b4ebf9a.pth)  | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_5shot-fine-tuning.log.json)  |
| [r101_fpn](/configs/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_10shot-fine-tuning.py) |        Y         |   3   |  10  |   77.0    |    57.9    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_10shot-fine-tuning_20211101_170749-f73f7a10.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_10shot-fine-tuning.log.json) |

**Note**:

- Following the original implementation, the contrastive loss only is added at VOC 3/5/10 shot setting, while in VOC 1/2 shot
  setting only the `fc_cls` and `fc_reg` layers are fine-tuned.
- Some arguments of configs are different from the official codes, for example, the official codes use aug test
  in some settings, while all the results reported above do not use `aug_test`.

## Results on COCO dataset

### Base Training

|                                     arch                                     | contrastive loss | Base mAP |                                                           ckpt(step1)                                                            |                                                                      ckpt(step2)                                                                       |                                                     log                                                     |
| :--------------------------------------------------------------------------: | :--------------: | :------: | :------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| [r101_fpn](/configs/detection/fsce/coco/fsce_r101_fpn_coco_base-training.py) |        N         |  39.50   | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_base-training_20211102_030413-a67975c7.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_base-training_20211102_030413_random-init-bbox-head-ea1c2981.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_base-training.log.json) |

### Few Shot Fine-tuning

|                                       arch                                        | shot | contrastive loss | Base mAP | Novel mAP |                                                                  ckpt                                                                   |                                                        log                                                         |
| :-------------------------------------------------------------------------------: | :--: | :--------------: | :------: | :-------: | :-------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| [r101_fpn](/configs/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning.py) |  10  |        N         |   31.7   |   11.7    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning_20211103_120353-3baa63b5.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/fsce/coco/fsce_r101_fpn_coco_30shot-fine-tuning.py) |  30  |        N         |   32.3   |   16.4    | [ckpt](https://download.openmmlab.com/mmfewshot/detection/fsce/coco/fsce_r101_fpn_coco_30shot-fine-tuning_20211103_140559-42edb8b2.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/fsce/coco/fsce_r101_fpn_coco_30shot-fine-tuning.log.json) |

**Note**:

- Some arguments of configs are different from the official codes, for example, the official codes use aug test
  in some settings, while all the results reported above do not use `aug_test`.
