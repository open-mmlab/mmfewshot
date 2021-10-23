"""Modified from https://github.com/wyharveychen/CloserLookFewShot and
https://github.com/RL-VIG/LibFewShot.

This file is only used in method maml for fast adaptation.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearWithFastWeight(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super().__init__(in_features, out_features)
        # Lazy hack to add fast weight link
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x: Tensor) -> Tensor:
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super().forward(x)
        return out


class Conv2dWithFastWeight(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple, str] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # Lazy hack to add fast weight link
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    None,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super().forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super().forward(x)

        return out


class BatchNorm2dWithFastWeight(nn.BatchNorm2d):

    def __init__(self, num_features: int) -> None:
        super().__init__(num_features)
        # Lazy hack to add fast weight link
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x: Tensor) -> Tensor:
        # batch_norm momentum hack: follow hack of Kate
        # Rakelly in pytorch-maml/src/layers.py
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1,
            )
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1,
            )
        return out


def convert_maml_module(module: nn.Module) -> nn.Module:
    """Convert a normal model to MAML model.

    Replace nn.Linear with LinearWithFastWeight, nn.Conv2d with
    Conv2dWithFastWeight and BatchNorm2d with BatchNorm2dWithFastWeight.

    Args:
        module(nn.Module): The module to be converted.

    Returns :
        nn.Module: A MAML module.
    """
    converted_module = module
    if isinstance(module, torch.nn.modules.Linear):
        converted_module = LinearWithFastWeight(
            module.in_features,
            module.out_features,
            False if module.bias is None else True,
        )
    elif isinstance(module, torch.nn.modules.Conv2d):
        converted_module = Conv2dWithFastWeight(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            False if module.bias is None else True,
        )
    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        converted_module = BatchNorm2dWithFastWeight(module.num_features)
    for name, child in module.named_children():
        converted_module.add_module(name, convert_maml_module(child))
    del module
    return converted_module
