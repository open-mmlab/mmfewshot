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
   - use all the images of base classes to train a base model.
   - use validation set to select the best model.
- **Step2: Meta Baseline Base training**
   - use all the images of base classes to train a base model with meta metric.
   - use validation set to select the best model.
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
  --options "load_from=work_dir/baseline_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth"

# meta testing
python ./tools/classification/test.py \
  configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py \
  work_dir/meta-baseline_conv4_1xb100_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:
- All the result are trained with single gpu.
- The base training of 1 shot and 5 shot use same training setting,
  but different validation setting.

## Results on CUB dataset of 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |

## Results on Mini-ImageNet dataset of 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |

## Results on Tiered-ImageNet of 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | - | -  | [ckpt]() | [log]() |
