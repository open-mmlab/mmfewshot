# Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector <a href="https://arxiv.org/abs/1908.01998"> (CVPR'2020)</a>


## Abstract

<!-- [ABSTRACT] -->

Conventional methods for object detection typically require
a substantial amount of training data and preparing
such high-quality training data is very labor-intensive. In
this paper, we propose a novel few-shot object detection
network that aims at detecting objects of unseen categories
with only a few annotated examples. Central to our method
are our Attention-RPN, Multi-Relation Detector and Contrastive
Training strategy, which exploit the similarity between
the few shot support set and query set to detect novel
objects while suppressing false detection in the background.
To train our network, we contribute a new dataset that contains
1000 categories of various objects with high-quality
annotations. To the best of our knowledge, this is one of the
first datasets specifically designed for few-shot object detection.
Once our few-shot network is trained, it can detect
objects of unseen categories without further training or finetuning.
Our method is general and has a wide range of potential
applications. We produce a new state-of-the-art performance
on different datasets in the few-shot setting. The
dataset link is https://github.com/fanq15/Few-Shot-Object-Detection-Dataset.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142847191-f9dd3254-bdcf-4c41-9b81-53fb0b332eb5.png" width="80%"/>
</div>



## Citation


<!-- [ALGORITHM] -->
```bibtex
@inproceedings{fan2020fsod,
    title={Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector},
    author={Fan, Qi and Zhuo, Wei and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={CVPR},
    year={2020}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.



## How to reproduce Attention RPN


Following the original implementation, it consists of 2 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Few shot fine-tuning**:
   - use the base model from step1 as model initialization and further fine tune the model with few shot datasets.


### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py 8

# step2: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_1shot-fine-tuning.py 8
```

**Note**:
- The default output path of base model in step1 is set to `work_dirs/{BASE TRAINING CONFIG}/latest.pth`.
  When the model is saved to different path, please update the argument `load_from` in step2 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.



## Results on VOC dataset

**Note**:
- The paper doesn't conduct experiments of VOC dataset.
  Therefore, we use the VOC setting of [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) to evaluate the method.
- Some implementation details should be noticed:
  - The training batch size are 8x2 for all the VOC experiments and 4x2 for all the COCO experiments(following the official repo).
  - Only the roi head will be trained during few shot fine-tuning for VOC experiments.
  - The iterations or training strategy for VOC experiments may not be the optimal.
- The performance of the base training and few shot setting can be unstable, even using the same random seed.
  To reproduce the reported few shot results, it is highly recommended using the released model for few shot fine-tuning.
- The difficult samples will not be used in base training or few shot setting.


### Base Training

| Arch  | Split | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py) | 1 | 71.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training_20211101_003606-58a8f413.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_base-training.py) | 2 | 73.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_base-training_20211101_040647-04570ae0.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_base-training.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_base-training.py) | 3 | 73.4 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_base-training_20211101_073701-5672bea8.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_base-training.log.json) |



### Few Shot Finetuning

| Arch  | Split | Shot | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: |:------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_1shot-fine-tuning.py) | 1 | 1 | 35.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_1shot-fine-tuning_20211107_224317-45e76f46.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_1shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_2shot-fine-tuning.py) | 1 | 2 | 36.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_2shot-fine-tuning_20211107_231154-e6209cb6.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_2shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_3shot-fine-tuning.py) | 1 | 3 | 39.1 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_3shot-fine-tuning_20211107_234134-ca895b22.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_3shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning.py) | 1 | 5 | 51.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning_20211108_001145-457dd542.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_10shot-fine-tuning.py) | 1 | 10 | 55.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_10shot-fine-tuning_20211108_004314-7c558c09.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_10shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_1shot-fine-tuning.py) | 2 | 1 |  20.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_1shot-fine-tuning_20211108_011609-87114fa4.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_1shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_2shot-fine-tuning.py) | 2 | 2 |  23.4 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_2shot-fine-tuning_20211108_014442-9043a914.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_2shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_3shot-fine-tuning.py) | 2 | 3 |  35.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_3shot-fine-tuning_20211102_004726-dfd9d7bb.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_3shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_5shot-fine-tuning.py) | 2 | 5 |  37.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_5shot-fine-tuning_20211102_011753-2ec1f244.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_5shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_10shot-fine-tuning.py) | 2 | 10 |  43.3 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_10shot-fine-tuning_20211102_015202-e914016b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_10shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_1shot-fine-tuning.py)  | 3 | 1 |  31.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_1shot-fine-tuning_20211102_022503-b47a5610.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_1shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_2shot-fine-tuning.py) | 3 | 2 |  30.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_2shot-fine-tuning_20211102_025331-7a4d4e9b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_2shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_3shot-fine-tuning.py) | 3 | 3 |  38.2 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_3shot-fine-tuning_20211102_032300-6a3c6fb4.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_3shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_5shot-fine-tuning.py) | 3 | 5 |  48.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_5shot-fine-tuning_20211102_035311-1420872c.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_5shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_10shot-fine-tuning.py) | 3 | 10 | 51.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_10shot-fine-tuning_20211102_042423-6724602a.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_10shot-fine-tuning.log.json) |



## Results on COCO dataset

**Note**:
- Following the original implementation, the training batch size are 4x2 for all the COCO experiments.
- The official implementation use different COCO data split from TFA, and we report the results of both setting.
  To reproduce the result following official data split (coco 17), please refer to [Data Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection/coco) to get more details about data preparation.
- The performance of the base training and few shot setting can be unstable, even using the same random seed.
  To reproduce the reported few shot results, it is highly recommended using the released model for few shot fine-tuning.

### Base Training

| Arch  | data source| Base mAP |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py) | [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) | 23.6 |[ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training_20211102_003348-da28cdfd.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.log.json) |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-base-training.py) | [official repo](https://github.com/fanq15/FewX/tree/master/datasets) | 24.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-base-training_20211102_003347-f9e2dab0.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-base-training.log.json) |


### Few Shot Finetuning

| Arch  | data source|  Shot | Novel mAP |  ckpt | log |
| :--------------: |  :--------------: | :-----------: |  :------: |:------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py) | [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) | 10 | 9.2 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning_20211103_003801-94ec8ada.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_30shot-fine-tuning.py) | [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) | 30 | 14.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_30shot-fine-tuning_20211103_010800-50611991.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_30shot-fine-tuning.log.json) |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py) | [official repo](https://github.com/fanq15/FewX/tree/master/datasets) | 10 | 11.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-10shot-fine-tuning_20211107_214729-6d046301.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-10shot-fine-tuning.log.json) |
