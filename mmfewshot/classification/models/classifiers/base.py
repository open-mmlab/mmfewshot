# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from mmcls.models.builder import (CLASSIFIERS, build_backbone, build_head,
                                  build_neck)
from mmcls.models.utils import Augments
from mmcv.runner import BaseModule
from torch import Tensor


@CLASSIFIERS.register_module()
class FewShotBaseClassifier(BaseModule):
    """Base class for classifier.

    Args:
        backbone (dict): Config of the backbone.
        neck (dict | None): Config of the neck. Default: None.
        head (dict | None): Config of classification head.
        train_cfg (dict | None): Training config. Default: None.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone: Dict,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)

        self.backbone = build_backbone(copy.deepcopy(backbone))

        if neck is not None:
            self.neck = build_neck(copy.deepcopy(neck))

        if head is not None:
            self.head = build_head(copy.deepcopy(head))

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

        self.meta_test_cfg = None
        # device_indicator is used to record runtime device
        self.register_buffer('device_indicator', torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.device_indicator.device

    def get_device(self):
        return self.device_indicator.get_device()

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def extract_feat(self, img: Tensor) -> Tensor:
        """Directly extract features from the backbone."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, **kwargs):
        """Forward Function."""

    @abstractmethod
    def forward_train(self, **kwargs):
        """Forward training data."""

    @abstractmethod
    def forward_support(self, **kwargs):
        """Forward support data in meta testing."""

    @abstractmethod
    def forward_query(self, **kwargs):
        """Forward query data in meta testing."""

    @staticmethod
    def _parse_losses(losses: Dict) -> Tuple[Dict, Dict]:
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer`): The optimizer of
                runner is passed to `train_step()`. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys:

                - `loss` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - `log_vars` contains all the variables to be sent to the
                  logger.
                - `num_samples` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @abstractmethod
    def before_meta_test(self, meta_test_cfg: Dict, **kwargs):
        """Used in meta testing.

        This function will be called before the meta testing.
        """

    @abstractmethod
    def before_forward_support(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """

    @abstractmethod
    def before_forward_query(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
