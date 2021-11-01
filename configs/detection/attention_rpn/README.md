<!-- [ALGORITHM] -->

#<summary><a href="https://arxiv.org/abs/1908.01998"> Attention RPN (CVPR'2020)</a></summary>

```bibtex
@inproceedings{fan2020fsod,
    title={Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector},
    author={Fan, Qi and Zhuo, Wei and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={CVPR},
    year={2020}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [here]() to get more details about the dataset and data preparation.


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
  When the model is saved to different path, please update the argument `load-from` in step2 few shot fine-tune configs instead
  of using `resume_from`.




## Results on VOC dataset

**Note**:
- The paper doesn't conduct experiments of VOC dataset.
  Therefore, we use the VOC setting of [TFA]() to evaluate the method.
- Something should be noticed:
  - The training batch size are 8x2 for all the VOC experiments.
  - Only the roi head will be trained during few shot fine-tuning.
  - The iterations or training strategy may not be the optimal.

### Base Training

| Arch  | Split | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py) | 1 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_base-training.py) | 2 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_base-training.py) | 3 | - | [ckpt]() | [log]() |

### Few Shot Finetuning

| Arch  | Split | Shot | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: |:------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_1shot-fine-tuning.py)  | 1 | 1 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_2shot-fine-tuning.py) | 1 | 2 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_3shot-fine-tuning.py) | 1 | 3 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning.py) | 1 | 5 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_10shot-fine-tuning.py) | 1 | 10 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_1shot-fine-tuning.py)  | 2 | 1 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_2shot-fine-tuning.py) | 2 | 2 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_3shot-fine-tuning.py) | 2 | 3 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_5shot-fine-tuning.py) | 2 | 5 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_10shot-fine-tuning.py) | 2 | 10 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_1shot-fine-tuning.py)  | 3 | 1 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_2shot-fine-tuning.py) | 3 | 2 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_3shot-fine-tuning.py) | 3 | 3 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_5shot-fine-tuning.py) | 3 | 5 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_10shot-fine-tuning.py) | 3 | 10 | - | [ckpt]() | [log]() |


## Results on COCO dataset

**Note**:
- Following the original implementation, the training batch size are 4x2 for all the COCO experiments.
- The official implementation use different COCO data split from TFA.
  Thus, we report the results of both setting.
- To reproduce the result following official data split, please refer to [here]() to get more details about data preparation.

### Base Training

| Arch  | data source| Base mAP |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py) | [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-base-training.py) | [official repo](https://github.com/fanq15/FewX/tree/master/datasets) | - | [ckpt]() | [log]() |

### Few Shot Finetuning

| Arch  | data source|  Shot | Novel mAP |  ckpt | log |
| :--------------: |  :--------------: | :-----------: |  :------: |:------: |:------: |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py) | [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) | 10 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_30shot-fine-tuning.py) | [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) | 30 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py) | [official repo](https://github.com/fanq15/FewX/tree/master/datasets) | 10 | - | [ckpt]() | [log]() |
