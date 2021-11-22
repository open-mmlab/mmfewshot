<!-- [ALGORITHM] -->

# <summary><a href="https://arxiv.org/abs/1904.04232"> Baseline (ICLR'2019)</a></summary>

```bibtex
@inproceedings{chen2019closerfewshot,
    title={A Closer Look at Few-shot Classification},
    author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
    booktitle={International Conference on Learning Representations},
    year={2019}
}
```

## How to Reproduce Baseline

It consists of two steps:
- **Step1: Base training**
   - use all the images of base classes to train a base model.
   - conduct meta testing on validation set to select the best model.

- **Step2: Meta Testing**:
   - use best model from step1, the best model are saved into `${WORK_DIR}/${CONFIG}/best_accuracy_mean.pth` in default.


### An example of CUB dataset with Conv4
```bash
# base training
python ./tools/classification/train.py \
  configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py \
  work_dir/baseline_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:
- All the result are trained with single gpu.
- The config of 1 shot and 5 shot use same training setting,
  but different meta test setting on validation set and test set.
- Currently, we use model selected by 1 shot validation (100 episodes) to
  evaluate both 1 shot and 5 shot setting on test set.
- The hyper-parameters in configs are roughly set and probably not the optimal one so
  feel free to tone and try different configurations.
  For example, try different learning rate or validation episodes for each setting.



## Results on CUB dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 47.73 | 0.41 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot_20211120_095923-3a346523.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot20211120_095923.log.json) |
| [conv4](/configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 68.77 | 0.38 | &uArr; | &uArr; |
| [resnet12](/configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 71.85 | 0.46 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot_20211120_095923-f1d13cf6.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot20211120_095923.log.json) |
| [resnet12](/configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 88.09 | 0.25  | &uArr; | &uArr; |

## Results on Mini-ImageNet dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 46.06 | 0.39 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot_20211120_095923-78b96fd7.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot.log.json) |
| [conv4](/configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 65.83 | 0.35 | &uArr; | &uArr; |
| [resnet12](/configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 60.0 | 0.44 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot_20211120_095923-26863e78.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 80.55 | 0.31  | &uArr; | &uArr; |
## Results on Tiered-ImageNet dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 45.49 | 0.41 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot_20211120_095923-e0b2cc06.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot.log.json) |
| [conv4](/configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 66.48 | 0.4 | &uArr; | &uArr; |
| [resnet12](/configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 66.11 | 0.49  | [ckpt](https://download.openmmlab.com/mmfewshot/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot_20211120_095923-17c5cf85.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 83.86 | 0.35 | &uArr; | &uArr; |
