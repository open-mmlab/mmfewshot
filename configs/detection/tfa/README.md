# Frustratingly Simple Few-Shot Object Detection <a href="https://arxiv.org/abs/2003.06957">(ICML'2020)</a>


## Abstract

<!-- [ABSTRACT] -->

Detecting rare objects from a few examples is an emerging problem.
Prior works show meta-learning is a promising approach.
But, fine-tuning techniques have drawn scant attention.
We find that fine-tuning only the last layer of existing detectors on rare classes is crucial to the few-shot object detection task.
Such a simple approach outperforms the meta-learning methods by roughly 2~20 points on current benchmarks and sometimes even doubles the accuracy of the prior methods.
However, the high variance in the few samples often leads to the unreliability of existing benchmarks.
We revise the evaluation protocols by sampling multiple groups of training examples to obtain stable comparisons and build new benchmarks based on three datasets: PASCAL VOC, COCO and LVIS.
Again, our fine-tuning approach establishes a new state of the art on the revised benchmarks.
The code as well as the pretrained models are available at https://github.com/ucbdrive/few-shot-object-detection.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142841882-4266e4a6-b93f-44d1-9754-72be2473d589.png" width="80%"/>
</div>



## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{wang2020few,
    title={Frustratingly Simple Few-Shot Object Detection},
    author={Wang, Xin and Huang, Thomas E. and  Darrell, Trevor and Gonzalez, Joseph E and Yu, Fisher}
    booktitle={International Conference on Machine Learning (ICML)},
    year={2020}
}
```



**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.


## How to reproduce TFA


Following the original implementation, it consists of 3 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Reshape the bbox head of base model**:
   - create a new bbox head for all classes fine-tuning (base classes + novel classes) using provided script.
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
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.


## Results on VOC dataset

### Base Training

| Arch  | Split | Base AP50 |  ckpt(step1) | ckpt(step2) | log |
| :------: | :-----------: | :------: | :------: | :------: |:------: |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py) | 1 | 80.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training_20211031_114821-efbd13e9.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training_20211031_114821_random-init-bbox-head-1e681852.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training.py) | 2 | 82.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training_20211031_114820-d47f8ef9.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training_20211031_114820_random-init-bbox-head-3d4c632c.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training.py) | 3 | 82.1 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training_20211031_114840-fd8a9864.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training_20211031_114840_random-init-bbox-head-9bb8c09b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training.log.json) |


**Note**:
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

| Arch  | Split | Shot | Base AP50 | Novel AP50 |  ckpt | log |
| :--------------: | :-----------: | :------: | :------: | :------: |:------: |:------: |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py)  | 1 | 1 | 79.2 | 41.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning_20211031_204528-9d6b2d28.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_2shot-fine-tuning.py)  | 1 | 2 | 79.2 | 49.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_2shot-fine-tuning_20211101_003504-d5083628.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_2shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py)  | 1 | 3 | 79.6 | 49.9 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning_20211101_005934-10ad61cd.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py)  | 1 | 5 | 79.6 | 58.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning_20211101_013516-5d682ebb.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_10shot-fine-tuning.py) | 1 | 10| 79.7 | 58.4 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_10shot-fine-tuning_20211101_023154-1f3d1ff1.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_10shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_1shot-fine-tuning.py)  | 2 | 1 | 80.3 | 26.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_1shot-fine-tuning_20211031_222829-a476e97f.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_1shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_2shot-fine-tuning.py)  | 2 | 2 | 78.1 | 30.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_2shot-fine-tuning_20211101_042109-eb35e87e.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_2shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_3shot-fine-tuning.py)  | 2 | 3 | 79.4 | 39.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_3shot-fine-tuning_20211101_044601-db0cd2b3.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_3shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_5shot-fine-tuning.py)  | 2 | 5 | 79.4 | 35.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_5shot-fine-tuning_20211101_052148-d2edde97.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_5shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_10shot-fine-tuning.py) | 2 | 10| 79.7 | 40.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_10shot-fine-tuning_20211101_061828-9c0cd7cd.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_10shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_1shot-fine-tuning.py)  | 3 | 1 | 80.5 | 34.0 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_1shot-fine-tuning_20211031_222908-509b00cd.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_1shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_2shot-fine-tuning.py)  | 3 | 2 | 80.6 | 39.3 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_2shot-fine-tuning_20211101_080733-2b0bc25b.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_2shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_3shot-fine-tuning.py)  | 3 | 3 | 81.1 | 42.8 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_3shot-fine-tuning_20211101_083226-48983379.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_3shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_5shot-fine-tuning.py)  | 3 | 5 | 80.8 | 51.4 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_5shot-fine-tuning_20211101_090812-da47ba99.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_5shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_10shot-fine-tuning.py) | 3 | 10| 80.7 | 50.6 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_10shot-fine-tuning_20211101_100435-ef8d4023.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_10shot-fine-tuning.log.json) |



## Results on COCO dataset

### Base Training

| Arch  | Base mAP |  ckpt(step1) |  ckpt(step2) | log |
| :------: | :-----------: | :------: |:------: |:------: |
| [r101_fpn](/configs/detection/tfa/coco/tfa_r101_fpn_coco_base-training.py) | 39.5 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_base-training_20211102_030413-a67975c7.pth) | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_base-training_20211102_030413_random-init-bbox-head-ea1c2981.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_base-training.log.json) |


### Few Shot Fine-tuning

| Arch  |  Shot | Base mAP | Novel mAP |  ckpt | log |
| :--------------: | :-----------: |  :------: |  :------: |:------: |:------: |
| [r101_fpn](/configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.py) | 10 | 35.2 | 10.4 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_20211102_162241-8abd2a82.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.log.json) |
| [r101_fpn](/configs/detection/tfa/coco/tfa_r101_fpn_coco_30shot-fine-tuning.py) | 30 | 36.7 | 14.7 | [ckpt](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_30shot-fine-tuning_20211103_001731-a63fce47.pth) | [log](https://download.openmmlab.com/mmfewshot/detection/tfa/coco/tfa_r101_fpn_coco_30shot-fine-tuning.log.json) |
