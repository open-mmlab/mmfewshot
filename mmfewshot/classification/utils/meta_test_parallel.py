import torch.nn as nn
from mmcv.parallel.scatter_gather import scatter_kwargs


class MetaTestParallel(nn.Module):
    """The MetaTestParallel module that supports DataContainer.

    Note that each task is tested on a single GPU. Thus the data and model
    on different GPU should be independent. :obj:`MMDistributedDataParallel`
    always automatically synchronizes the grad in different GPUs when doing
    the loss backward, which can not meet the requirements. Thus we simply
    copy the module and wrap it with an :obj:`MetaTestParallel`, which will
    send data to the device model.

    MetaTestParallel has two main differences with PyTorch DataParallel:

        - It supports a custom type :class:`DataContainer` which allows
          more flexible control of input data during both GPU and CPU
          inference.
        - It implement two more APIs ``before_meta_test()``,
          ``before_forward_support()`` and ``before_forward_query()``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, module: nn.Module, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.module = module
        self.device = self.module.device
        if self.device == 'cpu':
            self.device_id = [-1]
        else:
            self.device_id = [self.module.get_device()]

    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the datas in
        :class:`DataContainers` will still be gathered.
        """

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module(*inputs[0], **kwargs[0])

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def before_meta_test(self, *inputs, **kwargs) -> None:
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module.before_meta_test(*inputs[0], **kwargs[0])

    def before_forward_support(self, *inputs, **kwargs) -> None:
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module.before_forward_support(*inputs[0], **kwargs[0])

    def before_forward_query(self, *inputs, **kwargs) -> None:
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module.before_forward_query(*inputs[0], **kwargs[0])
