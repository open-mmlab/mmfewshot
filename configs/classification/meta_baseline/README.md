<!-- [ALGORITHM] -->

# <summary><a href="https://arxiv.org/abs/2003.04390"> Meta Baseline (ICCV'2021)</a></summary>

```bibtex
@inproceedings{chen2021meta,
    title={Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning},
    author={Chen, Yinbo and Liu, Zhuang and Xu, Huijuan and Darrell, Trevor and Wang, Xiaolong},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={9062--9071},
    year={2021}
}
```

## How to Reproduce Meta Baseline

It consists of three steps:
- **Step1: Baseline Base training**
   - use all the images of base classes to train a base model with linear head.
   - conduct meta testing on validation set to select the best model.
- **Step2: Meta Baseline Base training**
   - use all the images of base classes to train a base model with meta metric.
   - conduct meta testing on validation set to select the best model.
- **Step3: Meta Testing**:
   - use best model from step1.


### An example of CUB dataset with Conv4
```bash
# baseline base training
python ./tools/classification/train.py \
  configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py

# Meta Baseline base training
python ./tools/classification/train.py \
  configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py \
  --cfg-options load_from="work_dir/baseline_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth"

# meta testing
python ./tools/classification/test.py \
  configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py \
  work_dir/meta-baseline_conv4_1xb100_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:
- All the result are trained with single gpu.
- The configs of 1 shot and 5 shot use same training setting,
  but different meta test setting on validation set and test set.
- Currently, we use model selected by 1 shot validation (100 episodes) to
  evaluate both 1 shot and 5 shot setting on test set.
- The hyper-parameters in configs are roughly set and probably not the optimal one so
  feel free to tone and try different configurations.
  For example, try different learning rate or validation episodes for each setting.
  Anyway, we will continue to improve it.

## Results on CUB dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 58.98 | 0.47 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot_20211120_191622-bd94fc3c.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot20211120_191622.log.json) |
| [conv4](/configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 75.77 | 0.37 | &uArr; | &uArr; |
| [resnet12](/configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 78.16 | 0.43 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot_20211120_191622-8978c781.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot20211120_191622.log.json) |
| [resnet12](/configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 90.4 | 0.23 | &uArr; | &uArr; |

## Results on Mini-ImageNet dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 51.35 | 0.42 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot_20211120_191622-3ff1f837.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot20211120_191622.log.json) |
| [conv4](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 66.99 | 0.37 | &uArr; | &uArr; |
| [resnet12](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 64.53 | 0.45 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot_20211120_191622-70ecdc79.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot20211120_191622.log.json) |
| [resnet12](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 81.41 | 0.31 | &uArr; | &uArr; |

## Results on Tiered-ImageNet dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 53.09 | 0.48 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot_20211120_230843-e9c196e3.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot20211120_230843.log.json) |
| [conv4](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 67.85 | 0.43 | &uArr; | &uArr; |
| [resnet12](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 65.59 | 0.52 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot_20211120_230843-6f3a6e7e.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot20211120_230843.log.json) |
| [resnet12](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | 79.13 | 0.41 | &uArr; | &uArr; |
