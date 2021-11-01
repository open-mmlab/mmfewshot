<!-- [ALGORITHM] -->

#<summary><a href="https://arxiv.org/pdf/1909.13032.pdf"> Meta RCNN (ICCV'2019)</a></summary>

```bibtex
@inproceedings{yan2019meta,
    title={Meta r-cnn: Towards general solver for instance-level low-shot learning},
    author={Yan, Xiaopeng and Chen, Ziliang and Xu, Anni and Wang, Xiaoxi and Liang, Xiaodan and Lin, Liang},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    year={2019}
}
```
**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [here]() to get more details about the dataset and data preparation.

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
- The default output path of base model in step1 is set to `work_dirs/{BASE TRAINING CONFIG}/latest.pth`.
  When the model is saved to different path, please update the argument `load-from` in step2 few shot fine-tune configs instead
  of using `resume_from`.


## Results on VOC dataset

### Base Training

| Arch  | Split | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py) | 1 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_base-training.py) | 2 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_base-training.py) | 3 | - | [ckpt]() | [log]() |

### Few Shot Finetuning


| Arch  | Split | Shot | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: |:------: |:------: |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py)  | 1 | 1 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py) | 1 | 2 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py) | 1 | 3 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py) | 1 | 5 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py) | 1 | 10 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_1shot-fine-tuning.py)  | 1 | 1 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_2shot-fine-tuning.py) | 1 | 2 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_3shot-fine-tuning.py) | 1 | 3 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_5shot-fine-tuning.py) | 1 | 5 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_10shot-fine-tuning.py) | 1 | 10 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_1shot-fine-tuning.py)  | 1 | 1 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_2shot-fine-tuning.py) | 1 | 2 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_3shot-fine-tuning.py) | 1 | 3 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_5shot-fine-tuning.py) | 1 | 5 | - | [ckpt]() | [log]() |
| [r101 c4](/configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_10shot-fine-tuning.py) | 1 | 10 | - | [ckpt]() | [log]() |


## Results on COCO dataset
### Base Training

| Arch  | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: |:------: |
| [r50 c4](/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.py) | - | [ckpt]() | [log]() |

Few Shot Finetuning


| Arch  |  Shot | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: |  :------: |:------: |:------: |
| [r50 c4](/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.py) | 10 | - | [ckpt]() | [log]() |
| [r50 c4](/configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_30shot-fine-tuning.py) | 30 | - | [ckpt]() | [log]() |
