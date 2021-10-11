import numpy as np
import torch
from mmcls.models.builder import CLASSIFIERS

from mmfewshot.classification.datasets import label_wrapper
from mmfewshot.classification.models.utils import convert_maml_module
from .base import FewShotBaseClassifier


@CLASSIFIERS.register_module()
class MAMLClassifier(FewShotBaseClassifier):
    """Implementation of `MAML <https://arxiv.org/abs/1703.03400>`_.

    Args:
        num_inner_steps (int): Training steps for each task. Default: 5.
        inner_lr (float): Learning rate for each task. Default: 0.01.
        first_order (bool): First order approximation. Default: False.
    """

    def __init__(self,
                 num_inner_steps=5,
                 inner_lr=0.01,
                 first_order=False,
                 *args,
                 **kwargs):
        super(MAMLClassifier, self).__init__(*args, **kwargs)
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        self.first_order = first_order
        convert_maml_module(self)

    def forward(self,
                img=None,
                support_data=None,
                query_data=None,
                mode='train',
                **kwargs):
        """Calls one of (:func:`forward_train`, :func:`forward_query`,
        :func:`forward_support` and :func:`extract_feat`) according to
        the `mode`. The inputs of forward function would change with the
        `mode`.

        - When `mode` is 'train', the input will be query and support data
          for training.
        - When `mode` is any of 'support', 'query' and 'extract_feat', the
          input will be images.

        Args:
            img (Tensor): Used for func:`forward_query` or
                :func:`forward_support`. With shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
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
            assert (img is not None)
            return self.forward_query(img=img, **kwargs)
        elif mode == 'support':
            assert (img is not None)
            return self.forward_support(img=img, **kwargs)
        else:
            raise ValueError()

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined
        in this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
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

    def val_step(self, data, optimizer):
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

    def forward_train(self, support_data, query_data, **kwargs):
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
        support_img, query_img = support_data['img'], query_data['img']
        class_ids = torch.unique(support_data['gt_label']).cpu().tolist()
        np.random.shuffle(class_ids)
        support_label = label_wrapper(support_data['gt_label'], class_ids)
        query_label = label_wrapper(query_data['gt_label'], class_ids)

        self.fast_adapt(self.num_inner_steps, support_img, support_label)
        query_feats = self.extract_feat(query_img)
        loss = self.head.forward_train(query_feats, query_label)
        for weight in self.parameters():
            weight.fast = None
        return loss

    def forward_support(self, img, gt_label, **kwargs):
        """Forward support data in meta testing.

        Args:
            img (Tensor | None): With shape (N, C, H, W). Default: None.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        self.fast_adapt(self.meta_test_cfg['support']['num_inner_steps'], img,
                        gt_label)

    def forward_query(self, img, **kwargs):
        """Forward query data in meta testing.

        Args:
            img (Tensor | None): With shape (N, C, H, W). Default: None.

        Returns:
            list[np.ndarray]: A list of predicted results.
        """
        assert (img is not None)
        x = self.extract_feat(img)
        return self.head.forward_query(x)

    def fast_adapt(self, num_steps, img, labels):
        """Forward and update fast weight with input images and labels.

        Args:
            num_steps (int): The number of fast forward and update steps.
            img (Tensor): With shape (N, C, H, W).
            labels (Tensor): With shape (N).
        """
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        for step in range(num_steps):
            feats = self.extract_feat(img)
            inner_loss = self.head.forward_train(feats, labels)['loss']
            grads = torch.autograd.grad(
                inner_loss, fast_parameters, create_graph=True)
            fast_parameters = []
            if self.first_order:
                grads = [g.detach() for g in grads]
            for k, weight in enumerate(list(self.parameters())):
                if weight.fast is None:
                    weight.fast = weight - self.inner_lr * grads[k]
                else:
                    weight.fast = weight.fast - self.inner_lr * grads[k]
                fast_parameters.append(weight.fast)

    def before_meta_test(self, meta_test_cfg, **kwargs):
        """Used in meta testing.

        This function will be called before the meta testing.
        """
        self.meta_test_cfg = meta_test_cfg
        self.zero_grad()

    def before_forward_support(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        for weight in self.parameters():
            weight.fast = None
        self.backbone.train()
        self.head.train()

    def before_forward_query(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.backbone.eval()
        self.head.eval()
