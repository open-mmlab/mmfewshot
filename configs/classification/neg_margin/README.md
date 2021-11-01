<!-- [ALGORITHM] -->

# <summary><a href="https://arxiv.org/abs/2003.12060"> NegMargin (ECCV'2020)</a></summary>

```bibtex
@inproceedings{liu2020negative,
    title={Negative margin matters: Understanding margin in few-shot classification},
    author={Liu, Bin and Cao, Yue and Lin, Yutong and Li, Qi and Zhang, Zheng and Long, Mingsheng and Hu, Han},
    booktitle={European Conference on Computer Vision},
    pages={438--455},
    year={2020}
}
```

## How to Reproduce Neg-Margin

It consists of two steps:
- **Step1: Base training**
   - use all the images of base classes to train a base model.
   - use validation set to select the best model.

- **Step2: Meta Testing**:
   - use best model from step1.


### An example of CUB dataset with Conv4
```bash
# base training
python ./tools/classification/train.py \
  configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py \
  work_dir/neg-margin_cosine_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:
- All the result are trained with single gpu.
- The base training of 1 shot and 5 shot use same training setting,
  but different validation setting.

## Results on CUB dataset with 15 queries and 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |

## Results on Mini ImageNet dataset with 15 queries and 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |

## Results on Tiered ImageNet with 15 queries and 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | - | -  | [ckpt]() | [log]() |
