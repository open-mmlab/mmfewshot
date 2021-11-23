# Multi-Scale Positive Sample Refinement for Few-Shot Object Detection <a href="https://arxiv.org/abs/2007.09384"> (ECCV'2020)</a>
## Abstract

<!-- [ABSTRACT] -->
Few-shot object detection (FSOD) helps detectors adapt to unseen classes with few training instances, and is useful when manual
annotation is time-consuming or data acquisition is limited.
Unlike previous attempts that exploit few-shot classification techniques to facilitate FSOD, this work highlights the necessity of handling the problem of scale
variations, which is challenging due to the unique sample distribution.
To this end, we propose a Multi-scale Positive Sample Refinement (MPSR) approach to enrich object scales in FSOD.
It generates multi-scale positive samples as object pyramids and refines the prediction at various scales.
We demonstrate its advantage by integrating it as an auxiliary branch to the popular architecture of Faster R-CNN with FPN, delivering
a strong FSOD solution.
Several experiments are conducted on PASCAL VOC andMS COCO, and the proposed approach achieves state of the art results and significantly outperforms other counterparts, which shows its effectiveness.
Code is available at https://github.com/jiaxi-wu/MPSR.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142842616-f081795d-5dd4-4e8c-87b3-8a5192d4877c.png" width="80%"/>
</div>



## Citation

<!-- [ALGORITHM] -->
```bibtex
@inproceedings{wu2020mpsr,
  title={Multi-Scale Positive Sample Refinement for Few-Shot Object Detection},
  author={Wu, Jiaxi and Liu, Songtao and Huang, Di and Wang, Yunhong},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```


**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.

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
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.



## Results on VOC dataset

**Note**:
- We follow the official implementation using batch size 2x2 for training.
- The performance of few shot setting can be unstable, even using the same random seed.
  To reproduce the reported few shot results, it is highly recommended using the released model for few shot fine-tuning.
- The difficult samples will not be used in base training or few shot setting.

### Base Training

| Arch  | Split | Base AP50 |  ckpt | log |
| :------: | :-----------: | :------: | :------: |:------: |
| [r101 fpn](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py) | 1 | 80.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training_20211107_135130-ea747c7b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.log.json) |
| [r101 fpn](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.py) | 2 | 81.3 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training_20211107_135130-c7b4ee3f.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.log.json) |
| [r101 fpn](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.py) | 3 | 81.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training_20211107_135304-3528e346.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py) | 1 | 77.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth) | - |
| [r101 fpn*](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.py) | 2 | 78.3 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/convert_mpsr_r101_fpn_2xb2_voc-split2_base-training-1861c370.pth) | - |
| [r101 fpn*](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.py) | 3 | 77.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/convert_mpsr_r101_fpn_2xb2_voc-split3_base-training-1afa74d7.pth) | - |

**Note**:
- \* means the model is converted from [official repo](https://github.com/jiaxi-wu/MPSR), as we find that the base model trained from mmfewshot will
  get worse performance in fine-tuning especially in 1/2/3 shots, even their base training performance are higher.
  We will continue to investigate and improve it.


### Few Shot Fine-tuning


| Arch  | Split | Shot | Base AP50 | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: |:------: |:------: |:------: |
| [r101 fpn*](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py) | 1 | 1 | 60.6 | 38.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning_20211109_130330-444b743a.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_2shot-fine-tuning.py) | 1 | 2 | 65.9 | 45.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_2shot-fine-tuning_20211109_130330-3a778216.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_2shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_3shot-fine-tuning.py) | 1 | 3 | 68.1 | 49.2 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_3shot-fine-tuning_20211109_130347-f5baa2f7.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_3shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_5shot-fine-tuning.py) | 1 | 5 | 69.2 | 55.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_5shot-fine-tuning_20211109_130347-620065e8.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_5shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_10shot-fine-tuning.py) | 1 | 10 | 71.2 | 58.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_10shot-fine-tuning_20211109_130430-d87b3b4b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_10shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_1shot-fine-tuning.py) | 2 | 1 | 61.0 | 25.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_1shot-fine-tuning_20211107_195800-48163ea0.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_1shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_2shot-fine-tuning.py) | 2 | 2 | 66.9 | 29.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_2shot-fine-tuning_20211107_203755-65afa20b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_2shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_3shot-fine-tuning.py) | 2 | 3 | 67.6 | 40.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_3shot-fine-tuning_20211107_110120-832962b1.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_3shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_5shot-fine-tuning.py) | 2 | 5 | 70.4 | 41.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_5shot-fine-tuning_20211107_114449-ea834f31.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_5shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_10shot-fine-tuning.py) | 2 | 10 | 71.7 | 47.1 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_10shot-fine-tuning_20211107_122815-8108834b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_10shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_1shot-fine-tuning.py) | 3 | 1 | 57.9 | 34.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_1shot-fine-tuning_20211107_131308-c0e1d1f0.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_1shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_2shot-fine-tuning.py) | 3 | 2 | 65.7 | 41.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_2shot-fine-tuning_20211107_135527-70053e26.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_2shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_3shot-fine-tuning.py) | 3 | 3 | 69.1 | 44.1 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_3shot-fine-tuning_20211107_155433-8955b1d3.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_3shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_5shot-fine-tuning.py) | 3 | 5 | 70.4 | 48.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_5shot-fine-tuning_20211107_171449-a9931117.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_5shot-fine-tuning.log.json) |
| [r101 fpn*](/configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_10shot-fine-tuning.py) | 3 | 10 | 72.5 | 51.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_10shot-fine-tuning_20211107_183534-698b6503.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_10shot-fine-tuning.log.json) |

- \* means using base model converted from [official repo](https://github.com/jiaxi-wu/MPSR)

## Results on COCO dataset

**Note**:
- We follow the official implementation using batch size 2x2 for training.
- The performance of base training and few shot setting can be unstable, even using the same random seed.
  To reproduce the reported few shot results, it is highly recommended using the released model for few shot fine-tuning.


### Base Training

| Arch  | Base mAP |  ckpt | log |
| :------: | :-----------: | :------: |:------: |
| [r101 fpn](/configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_base-training.py) | 34.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_base-training_20211103_164720-c6998b36.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_base-training.log.json) |

### Few Shot Fine-tuning


| Arch  |  Shot | Base mAP | Novel mAP |  ckpt | log |
| :--------------: | :-----------: |  :------: |:------: |:------: |:------: |
| [r101 fpn](/configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_10shot-fine-tuning.py) | 10 | 23.2 | 12.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_10shot-fine-tuning_20211104_161345-c4f1955a.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_10shot-fine-tuning.log.json) |
| [r101 fpn](/configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_30shot-fine-tuning.py) | 30 | 25.2 | 18.1 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_30shot-fine-tuning_20211104_161611-fedc6a63.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_30shot-fine-tuning.log.json) |
