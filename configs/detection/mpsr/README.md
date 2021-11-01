<!-- [ALGORITHM] -->

#<summary><a href="https://arxiv.org/abs/2007.09384"> MPSR (ECCV'2020)</a></summary>

```bibtex
@inproceedings{wu2020mpsr,
  title={Multi-Scale Positive Sample Refinement for Few-Shot Object Detection},
  author={Wu, Jiaxi and Liu, Songtao and Huang, Di and Wang, Yunhong},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```
**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [here]() to get more details about the dataset and data preparation.

## How to reproduce MPSR
Following the original implementation, it consists of 2 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Few shot fine-tuning**:
   - use the base model from step1 as model initialization and further fine tune the model with few shot datasets.


### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py 8

# step2: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py 8
```

**Note**:
- The default output path of base model in step1 is set to `work_dirs/{BASE TRAINING CONFIG}/latest.pth`.
  When the model is saved to different path, please update the argument `load-from` in step2 few shot fine-tune configs instead
  of using `resume_from`.

## Results on VOC dataset

### Base Training

| Arch  | Split | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py) | 1 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.py) | 2 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.py) | 3 | - | [ckpt]() | [log]() |

### Few Shot Fine-tuning


| Arch  | Split | Shot | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: |:------: |:------: |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py)  | 1 | 1 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_2shot-fine-tuning.py) | 1 | 2 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_3shot-fine-tuning.py) | 1 | 3 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_5shot-fine-tuning.py) | 1 | 5 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_10shot-fine-tuning.py) | 1 | 10 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_1shot-fine-tuning.py)  | 2 | 1 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_2shot-fine-tuning.py) | 2 | 2 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_3shot-fine-tuning.py) | 2 | 3 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_5shot-fine-tuning.py) | 2 | 5 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_10shot-fine-tuning.py) | 2 | 10 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_1shot-fine-tuning.py)  | 3 | 1 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_2shot-fine-tuning.py) | 3 | 2 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_3shot-fine-tuning.py) | 3 | 3 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_5shot-fine-tuning.py) | 3 | 5 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_10shot-fine-tuning.py) | 3 | 10 | - | [ckpt]() | [log]() |


## Results on COCO dataset
### Base Training

| Arch  | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: |:------: |
| [r101 fpn](/configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_base-training.py) | - | [ckpt]() | [log]() |

### Few Shot Fine-tuning


| Arch  |  Shot | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: |  :------: |:------: |:------: |
| [r101 fpn](/configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_10shot-fine-tuning.py)  | 10 | - | [ckpt]() | [log]() |
| [r101 fpn](/configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_30shot-fine-tuning.py) | 30 | - | [ckpt]() | [log]() |
