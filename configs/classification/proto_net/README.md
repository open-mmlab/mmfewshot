<!-- [ALGORITHM] -->

# <summary><a href="https://arxiv.org/abs/1703.05175"> ProtoNet (NeurIPS'2017)</a></summary>

```bibtex
@inproceedings{snell2017prototypical,
    title={Prototypical networks for few-shot learning},
    author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
    booktitle={Proceedings of the 31st International Conference on Neural Information Processing Systems},
    pages={4080--4090},
    year={2017}
}
```
## How to Reproduce ProtoNet

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
  configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py

# meta testing
python ./tools/classification/test.py \
  configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py \
  work_dir/proto-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth
```

**Note**:
- All the result are trained with single gpu.
- The base training of 1 shot and 5 shot use same training setting,
  but different validation setting.

## Results on CUB dataset of 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |

## Results on Mini-ImageNet dataset of 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | - | [ckpt]() | [log]() |
| [resnet12](/configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |

## Results on Tiered-ImageNet of 1000 episodes

| Arch  | Input Size | Batch Size | way | shot | mean Acc | std | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [conv4](/configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot.py)  | 84x84 | 64 | 5  | 1 | - | - | [ckpt]() | [log]() |
| [conv4](/configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 |  - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py) | 84x84 | 64 | 5 | 1 | - | -  | [ckpt]() | [log]() |
| [resnet12](/configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py) | 84x84 | 64 | 5 | 5 | - | -  | [ckpt]() | [log]() |
