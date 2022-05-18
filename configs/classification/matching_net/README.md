# Matching Networks for One Shot Learning <a href="https://arxiv.org/abs/1606.04080"> (NeurIPS'2016)</a>

## Abstract

<!-- [ABSTRACT] -->

Learning from a few examples remains a key challenge in machine learning.
Despite recent advances in important domains such as vision and language, the
standard supervised deep learning paradigm does not offer a satisfactory solution
for learning new concepts rapidly from little data. In this work, we employ ideas
from metric learning based on deep neural features and from recent advances
that augment neural networks with external memories. Our framework learns a
network that maps a small labelled support set and an unlabelled example to its
label, obviating the need for fine-tuning to adapt to new class types. We then define
one-shot learning problems on vision (using Omniglot, ImageNet) and language
tasks. Our algorithm improves one-shot accuracy on ImageNet from 87.6% to
93.2% and from 88.0% to 93.8% on Omniglot compared to competing approaches.
We also demonstrate the usefulness of the same model on language modeling by
introducing a one-shot task on the Penn Treebank.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142851095-6899580d-3584-442e-b1f2-6cea844e1532.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{vinyals2016matching,
    title={Matching networks for one shot learning},
    author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Tim and Wierstra, Daan and others},
    booktitle={Advances in Neural Information Processing Systems},
    pages={3630--3638},
    year={2016}
}
```

## How to Reproduce MatchingNet

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
  configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py \
  work_dir/matching-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth
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
- The training batch size is calculated by `num_support_way` * (`num_support_shots` + `num_query_shots`)

## Results on CUB dataset with 2000 episodes

| Arch                                                                                                | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                           ckpt                                                                            |                                                                 log                                                                  |
| :-------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  63.65   | 0.5  |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot_20211120_100611-dfc09deb.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.log.json)   |
| [conv4](/configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  76.88   | 0.39 |                                                                             ⇑                                                                             |                                                                  ⇑                                                                   |
| [resnet12](/configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  78.33   | 0.45 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot_20211120_100611-d396459d.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.log.json) |
| [resnet12](/configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-5shot.py) |   84x84    |    105     |  5  |  5   |  88.98   | 0.26 |                                                                             ⇑                                                                             |                                                                  ⇑                                                                   |

## Results on Mini-ImageNet dataset with 2000 episodes

| Arch                                                                                                                    | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                                     ckpt                                                                                      |                                                                           log                                                                            |
| :---------------------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  53.35   | 0.44 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot_20211120_100611-cfc24845.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |   66.3   | 0.38 |                                                                                       ⇑                                                                                       |                                                                            ⇑                                                                             |
| [resnet12](/configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |   59.3   | 0.45 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211120_100611-62e83016.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |  72.63   | 0.36 |                                                                                       ⇑                                                                                       |                                                                            ⇑                                                                             |

## Results on Tiered-ImageNet dataset with 2000 episodes

| Arch                                                                                                                        | Input Size | Batch Size | way | shot | mean Acc | std  |                                                                                       ckpt                                                                                        |                                                                             log                                                                              |
| :-------------------------------------------------------------------------------------------------------------------------- | :--------: | :--------: | :-: | :--: | :------: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [conv4](/configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot.py)       |   84x84    |    105     |  5  |  1   |  48.20   | 0.48 |  [ckpt](https://download.openmmlab.com/mmfewshot/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot_20211120_100611-e70e9548.pth)   |  [log](https://download.openmmlab.com/mmfewshot/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot.log.json)   |
| [conv4](/configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-5shot.py)       |   84x84    |    105     |  5  |  5   |  61.19   | 0.43 |                                                                                         ⇑                                                                                         |                                                                              ⇑                                                                               |
| [resnet12](/configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py) |   84x84    |    105     |  5  |  1   |  58.97   | 0.52 | [ckpt](https://download.openmmlab.com/mmfewshot/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211120_100611-90c3124c.pth) | [log](https://download.openmmlab.com/mmfewshot/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot.log.json) |
| [resnet12](/configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py) |   84x84    |    105     |  5  |  5   |   72.1   | 0.45 |                                                                                         ⇑                                                                                         |                                                                              ⇑                                                                               |
