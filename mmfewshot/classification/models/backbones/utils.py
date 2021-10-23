import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli

# This part of code is modified from https://github.com/kjunelee/MetaOptNet


class DropBlock(nn.Module):

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: Tensor, gamma: float) -> Tensor:
        # Randomly zeroes 2D spatial blocks of the input tensor.
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1),
                 width - (self.block_size - 1)))
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size(
            )[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask: Tensor) -> Tensor:
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        non_zero_idxes = mask.nonzero()
        nr_blocks = non_zero_idxes.shape[0]

        offsets = torch.stack([
            torch.arange(self.block_size).view(-1, 1).expand(
                self.block_size, self.block_size).reshape(-1),
            torch.arange(self.block_size).repeat(self.block_size),
        ]).t()
        offsets = torch.cat(
            (torch.zeros(self.block_size**2, 2).long(), offsets.long()), 1)
        offsets = offsets.to(mask.device)

        if nr_blocks > 0:
            non_zero_idxes = non_zero_idxes.repeat(self.block_size**2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxes = non_zero_idxes + offsets
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxes[:, 0], block_idxes[:, 1],
                        block_idxes[:, 2], block_idxes[:, 3]] = 1.
        else:
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask
        return block_mask
