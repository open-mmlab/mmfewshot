from mmcv.runner import HOOKS, Hook
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel


@HOOKS.register_module()
class ContrastiveLossDecayHook(Hook):
    """Hook for contrast loss weight decay used in FSCE.

    Args:
        decay_steps (list[int] | tuple[int]): Each item in the list is
            the step to decay the loss weight.
        decay_rate (float): Decay rate. Default: 0.5.
    """

    def __init__(self, decay_steps, decay_rate=0.5):
        assert isinstance(decay_steps, (list, tuple)), \
            '`decay_steps` should be list or tuple.'
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def before_iter(self, runner):
        runner_iter = runner.iter + 1
        decay_rate = 1.0
        for step in self.decay_steps:
            if runner_iter > step:
                decay_rate *= self.decay_rate
        if isinstance(runner.model, DataParallel) or \
                isinstance(runner.model, DistributedDataParallel):
            runner.model.module.roi_head.bbox_head.set_decay_rate(decay_rate)
        else:
            runner.model.roi_head.bbox_head.set_decay_rate(decay_rate)
