import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS

from mmfewshot.classification.datasets import label_wrapper
from .base_head import FewShotBaseHead


@HEADS.register_module()
class MetaBaselineHead(FewShotBaseHead):
    """Classification head for `MetaBaseline
    https://arxiv.org/abs/2003.04390`_.

    Args:
        temperature (float): Scaling factor of `cls_score`. Default: 10.0.
        learnable_temperature (bool): Whether to use learnable scale factor
            or not. Default: True.
    """

    def __init__(self,
                 temperature=10.0,
                 learnable_temperature=True,
                 *args,
                 **kwargs):
        super(MetaBaselineHead, self).__init__(*args, **kwargs)
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        self.support_feats = []
        self.support_labels = []
        self.mean_support_feats = None
        self.class_ids = None

    def forward_train(self, support_feats, support_labels, query_feats,
                      query_labels, **kwargs):
        """Forward training data.

        Args:
            support_feats (Tensor): Features of support data with shape (N, C).
            support_labels (Tensor): Labels of support data with shape (N).
            query_feats (Tensor): Features of query data with shape (N, C).
            query_labels (Tensor): Labels of query data with shape (N).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        class_ids = torch.unique(support_labels).cpu().tolist()
        mean_support_feats = torch.cat([
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in class_ids
        ],
                                       dim=0)
        cosine_distance = torch.mm(
            F.normalize(query_feats),
            F.normalize(mean_support_feats).transpose(0, 1))
        scores = cosine_distance * self.temperature
        query_labels = label_wrapper(query_labels, class_ids)
        losses = self.loss(scores, query_labels)
        return losses

    def forward_support(self, x, gt_label, **kwargs):
        """Forward support data in meta testing."""
        self.support_feats.append(x)
        self.support_labels.append(gt_label)

    def forward_query(self, x, **kwargs):
        """Forward query data in meta testing."""
        cosine_distance = torch.mm(
            F.normalize(x),
            F.normalize(self.mean_support_feats).transpose(0, 1))
        scores = cosine_distance * self.temperature
        pred = F.softmax(scores, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # reset saved features for testing new task
        self.support_feats.clear()
        self.support_labels.clear()
        self.class_ids = None
        self.mean_support_feats = None

    def before_forward_query(self):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        support_feats = torch.cat(self.support_feats, dim=0)
        support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids, _ = torch.unique(support_labels).sort()
        self.mean_support_feats = torch.cat([
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in self.class_ids
        ],
                                            dim=0)
