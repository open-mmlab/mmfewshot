from mmcv.parallel.scatter_gather import scatter_kwargs
from torch.nn import Module


class DeviceWrapper(Module):
    """The DeviceWrapper module that supports DataContainer.

    DeviceWrapper has two main differences with PyTorch DataParallel:

        - It supports a custom type :class:`DataContainer` which allows
          more flexible control of input data during both GPU and CPU
          inference.
        - It implement two more APIs ``before_meta_test()``,
          ``before_forward_support()`` and ``before_forward_query()``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, module, dim=0):
        super(DeviceWrapper, self).__init__()
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

    def before_meta_test(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module.before_meta_test(*inputs[0], **kwargs[0])

    def before_forward_support(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module.before_forward_support(*inputs[0], **kwargs[0])

    def before_forward_query(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_id)
        if not inputs and not kwargs:
            inputs = ((), )
            kwargs = ({}, )
        return self.module.before_forward_query(*inputs[0], **kwargs[0])
