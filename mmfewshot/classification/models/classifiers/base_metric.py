# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmcls.models.builder import CLASSIFIERS
from torch import Tensor
from typing_extensions import Literal

from .base import BaseFewShotClassifier


@CLASSIFIERS.register_module()
class BaseMetricClassifier(BaseFewShotClassifier):
    """Base class for meta metric based classifier."""

    def forward(self,
                img: Tensor = None,
                feats: Tensor = None,
                support_data: Dict = None,
                query_data: Dict = None,
                mode: Literal['train', 'support', 'query',
                              'extract_feat'] = 'train',
                **kwargs) -> Union[Dict, List, Tensor]:
        """Calls one of (:func:`forward_train`, :func:`forward_query`,
        :func:`forward_support` and :func:`extract_feat`) according to
        the `mode`. The inputs of forward function would change with the
        `mode`.

        - When `mode` is 'train', the input will be query and support data
          for training.
        - When `mode` is 'support' or 'query', the input will be either
          images or extracted features.
        - When `mode` is 'extract_feat', the input will be images.

        Args:
            img (Tensor): Used for func:`forward_query` or
                :func:`forward_support`. With shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
                Default: None.
            feats (Tensor): Used for func:`forward_query` or
                :func:`forward_support`. With shape (N, C, H, W) or (N, C).
                Default: None.
            query_data (dict): Used for :func:`forward_train`. Dict of
                query data and data info where each dict has: `img`,
                `img_metas`, `gt_labels`. Default: None.
            support_data (dict): Used for :func:`forward_train`. Dict of
                query data and data info where each dict has: `img`,
                `img_metas`, `gt_labels`. Default: None.
            mode (str): Indicate which function to call. Options are 'train',
                'support', 'query' and 'extract_feat'. Default: 'train'.
        """

        if mode == 'train':
            assert (support_data is not None) and (query_data is not None)
            return self.forward_train(
                support_data=support_data, query_data=query_data, **kwargs)
        elif mode == 'query':
            assert (img is not None) or (feats is not None)
            return self.forward_query(img=img, feats=feats, **kwargs)
        elif mode == 'support':
            assert (img is not None) or (feats is not None)
            return self.forward_support(img=img, feats=feats, **kwargs)
        elif mode == 'extract_feat':
            assert img is not None
            return self.extract_feat(img=img)
        else:
            raise ValueError()

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
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['support_data']['img']))

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
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['support_data']['img']))

        return outputs

    def forward_train(self, support_data: Dict, query_data: Dict,
                      **kwargs) -> Dict:
        """Forward computation during training.

        Args:
            query_data (dict): Used for :func:`forward_train`. Dict of
                query data and data info where each dict has: `img`,
                `img_metas`, `gt_labels`. Default: None.
            support_data (dict): Used for :func:`forward_train`. Dict of
                query data and data info where each dict has: `img`,
                `img_metas`, `gt_labels`. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if support_data.get('feats', None):
            support_feats = support_data['feats']
        else:
            support_img = support_data['img']
            support_feats = self.extract_feat(support_img)

        if query_data.get('feats', None):
            query_feats = query_data['feats']
        else:
            query_img = query_data['img']
            query_feats = self.extract_feat(query_img)

        losses = dict()
        loss = self.head.forward_train(support_feats, support_data['gt_label'],
                                       query_feats, query_data['gt_label'])
        losses.update(loss)

        return losses

    def forward_support(self,
                        gt_label: Tensor,
                        img: Optional[Tensor] = None,
                        feats: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward support data in meta testing.

        Input can be either images or extracted features.

        Args:
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images.
            img (Tensor | None): With shape (N, C, H, W). Default: None.
            feats (Tensor | None): With shape (N, C).

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert (img is not None) or (feats is not None)
        if feats is None:
            x = self.extract_feat(img)
        else:
            x = feats
        return self.head.forward_support(x, gt_label)

    def forward_query(self,
                      img: Tensor = None,
                      feats: Tensor = None,
                      **kwargs) -> List:
        """Forward query data in meta testing.

        Input can be either images or extracted features.

        Args:
            img (Tensor | None): With shape (N, C, H, W). Default: None.
            feats (Tensor | None): With shape (N, C).

        Returns:
            list[np.ndarray]: A list of predicted results.
        """
        assert (img is not None) or (feats is not None)
        if feats is None:
            x = self.extract_feat(img)
        else:
            x = feats
        return self.head.forward_query(x)

    def before_meta_test(self, meta_test_cfg: Dict, **kwargs) -> None:
        """Used in meta testing.

        This function will be called before the meta testing.
        """
        # For each test task the model will be copied and reset.
        # When using extracted features to accelerate meta testing,
        # the unused backbone will be removed to avoid copying
        # useless parameters.
        if meta_test_cfg.get('fast_test', False):
            self.backbone = None
        else:
            # fix backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.meta_test_cfg = meta_test_cfg

    def before_forward_support(self, **kwargs) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        self.head.before_forward_support()

    def before_forward_query(self, **kwargs) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.head.before_forward_query()
