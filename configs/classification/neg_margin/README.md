# Negative Margin Matters: Understanding Margin in Few-shot Classification <a href="https://arxiv.org/abs/2003.12060"> (ECCV'2020)</a>

## Abstract

<!-- [ABSTRACT] -->

This paper introduces a negative margin loss to metric learning based few-shot learning methods. The negative margin loss significantly outperforms regular softmax loss, and achieves state-of-the-art
accuracy on three standard few-shot classification benchmarks with few
bells and whistles. These results are contrary to the common practice
in the metric learning field, that the margin is zero or positive. To understand why the negative margin loss performs well for the few-shot
classification, we analyze the discriminability of learned features w.r.t
different margins for training and novel classes, both empirically and
theoretically. We find that although negative margin reduces the feature discriminability for training classes, it may also avoid falsely mapping samples of the same novel class to multiple peaks or clusters, and
thus benefit the discrimination of novel classes. Code is available at
https://github.com/bl0/negative-margin.few-shot.


<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142850190-c7ceab7e-468a-48ad-b8f5-e2d0e2df13ae.png" width="80%"/>
</div>



## Citation

<!-- [ALGORITHM] -->
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
   - conduct meta testing on validation set to select the best model.

- **Step2: Meta Testing**:
   - use best model from step1, the best model are saved into `${WORK_DIR}/${CONFIG}/best_accuracy_mean.pth` in default.



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
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 64.08 | 0.48 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot_20211120_100620-5415a152.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.log.json) |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | 80.69 | 0.34 | &uArr; | &uArr; |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 78.54 | 0.46 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot_20211120_100620-b4ab9cc1.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot.log.json) |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | 90.19 | 0.24 | &uArr; | &uArr; |

## Results on Mini ImageNet dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 51.15 | 0.42 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot_20211120_104933-8a1340d3.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot.log.json) |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | 67.32 | 0.37 | &uArr; | &uArr; |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 61.7 | 0.46 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot_20211120_110018-e3aae9b5.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | 78.03 | 0.33 | &uArr; | &uArr; |

## Results on Tiered ImageNet dataset with 2000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | 54.07 | 0.49 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot_20211120_110238-7a81ac2a.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot.log.json) |
| [conv4](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | 70.25 | 0.41 | &uArr; | &uArr; |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | 65.88 | 0.53  | [ckpt](https://download.openmmlab.com/mmfewshot/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot_20211120_111912-344342e6.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py) | 84x84 | 64 | 5 | 5 | 81.06 | 0.39  | &uArr; | &uArr; |
