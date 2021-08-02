import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads import ConvFCBBoxHead


@HEADS.register_module()
class ContrastiveBBoxHead(ConvFCBBoxHead):
    """BBoxHead for `FSCE <https://arxiv.org/abs/2103.05950>`_.

    Args:
        mlp_head_channels (int): Output channels of contrast branch
            mlp. Default: 128.
        with_weight_decay (bool): Whether to decay loss weight. Default: False.
        loss_contrast (dict): Config of contrast loss.
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 mlp_head_channels=128,
                 with_weight_decay=False,
                 loss_contrast=dict(
                     type='SupervisedContrastiveLoss',
                     temperature=0.1,
                     iou_threshold=0.5,
                     loss_weight=1.0,
                     reweight_type='none'),
                 scale=20,
                 learnable_scale=False,
                 eps=1e-5,
                 *args,
                 **kwargs):
        super(ContrastiveBBoxHead, self).__init__(*args, **kwargs)
        # override the fc_cls in :obj:`ConvFCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.mlp_head_channels = mlp_head_channels
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        self.contrast_loss = build_loss(copy.deepcopy(loss_contrast))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
                contrast_feat (Tensor): Box features for contrast loss,
                    has shape (num_proposals, C).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x
        x_contra = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # reg branch
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # cls branch
        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None

        # contrastive branch
        contrast_feat = self.contrastive_head(x_contra)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        return cls_score, bbox_pred, contrast_feat

    def set_decay_rate(self, decay_rate):
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    @force_fp32(apply_to=('contrast_feat'))
    def loss_contrast(self,
                      contrast_feat,
                      proposal_ious,
                      labels,
                      reduction_override=None):
        """Loss for contract.

        Args:
            contrast_feat (tensor): BBox features with shape (N, C)
                used for contrast loss.
            proposal_ious (tensor): IoU between proposal and ground truth
                corresponding to each BBox features with shape (N).
            labels (tensor): Labels for each BBox features with shape (N).
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """

        losses = dict()
        if self.with_weight_decay:
            decay_rate = self._decay_rate
        else:
            decay_rate = None
        losses['loss_contrast'] = self.contrast_loss(
            contrast_feat,
            labels,
            proposal_ious,
            decay_rate=decay_rate,
            reduction_override=reduction_override)
        return losses
