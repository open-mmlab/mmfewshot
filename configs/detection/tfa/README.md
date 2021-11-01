<!-- [ALGORITHM] -->

# <summary><a href="https://arxiv.org/abs/2003.06957"> TFA (ICML'2020)</a></summary>

```bibtex
@inproceedings{wang2020few,
    title={Frustratingly Simple Few-Shot Object Detection},
    author={Wang, Xin and Huang, Thomas E. and  Darrell, Trevor and Gonzalez, Joseph E and Yu, Fisher}
    booktitle={International Conference on Machine Learning (ICML)},
    year={2020}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [here]() to get more details about the dataset and data preparation.


## How to reproduce TFA


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
    configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py 8

# step2: reshape the bbox head of base model for few shot fine-tuning
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth \
    --method randinit \
    --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training

# step3: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py 8
```

**Note**:
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load-from` in step3 few shot fine-tune configs instead
  of using `resume_from`.



## Results on VOC dataset

### Base Training

| Arch  | Split | Base AP50 |  ckpt(step1) | ckpt(step2) | log |
| :------: | :-----------: | :------: | :------: | :------: |:------: |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py) | 1 | - | [ckpt]() | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training.py) | 2 | - | [ckpt]() | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training.py) | 3 | - | [ckpt]() | [ckpt]() | [log]() |

**Note**:
- The performance of the same few shot setting using different base training models can be dramatically unstable
  (AP50 can fluctuate by 5.0 or more), even their mAP on base classes are very close.
- Temporally, the solution to getting a good base model is training the base model with different random seed.
  Also, the random seed used in this code base may not the optimal one, and it is possible to get the higher results by using
  other random seeds.
  However, using the same random seed still can not guarantee the identical result each time.

- To reproduce the reported few shot results, it is highly recommended using the released step2 model for few shot fine-tuning.
  We will continue to investigate and improve it.


### Few Shot Fine-tuning

| Arch  | Split | Shot | Base AP50 | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: | :------: |:------: |:------: |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py)  | 1 | 1 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_2shot-fine-tuning.py)  | 1 | 2 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py)  | 1 | 3 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py)  | 1 | 5 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_10shot-fine-tuning.py) | 1 | 10| - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_1shot-fine-tuning.py)  | 2 | 1 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_2shot-fine-tuning.py)  | 2 | 2 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_3shot-fine-tuning.py)  | 2 | 3 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_5shot-fine-tuning.py)  | 2 | 5 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_10shot-fine-tuning.py) | 2 | 10| - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_1shot-fine-tuning.py)  | 3 | 1 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_2shot-fine-tuning.py)  | 3 | 2 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_3shot-fine-tuning.py)  | 3 | 3 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_5shot-fine-tuning.py)  | 3 | 5 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_10shot-fine-tuning.py) | 3 | 10| - | - | [ckpt]() | [log]() |


## Results on COCO dataset

### Base Training

| Arch  | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: |:------: |
| [r101_fpn](/configs/detection/tfa/coco/tfa_r101_fpn_coco_base-training.py) | - | - | [ckpt]() | [log]() |

### Few Shot Fine-tuning

| Arch  |  Shot | Base mAP | Novel mAP |  ckpt | log |
| :--------------: | :-----------: |  :------: |  :------: |:------: |:------: |
| [r101_fpn](/configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.py)  | 10 | - | - | [ckpt]() | [log]() |
| [r101_fpn](/configs/detection/tfa/coco/tfa_r101_fpn_coco_30shot-fine-tuning.py)  | 30 | - | - | [ckpt]() | [log]() |
