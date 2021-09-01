import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS

from mmfewshot.classification.datasets import label_wrapper
from .base_head import FewShotBaseHead


@HEADS.register_module()
class PrototypeHead(FewShotBaseHead):
    """Classification head for `ProtoNet  <https://arxiv.org/abs/1703.05175>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, *args, **kwargs):
        super(PrototypeHead, self).__init__(*args, **kwargs)
        self.support_feats = []
        self.support_labels = []
        self.prototype_feats = None

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
        prototype_feats = [
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in class_ids
        ]
        prototype_feats = torch.cat(prototype_feats, dim=0)

        cls_scores = -1 * torch.cdist(
            query_feats.unsqueeze(0), prototype_feats.unsqueeze(0)).squeeze(0)
        query_labels = label_wrapper(query_labels, class_ids)
        losses = self.loss(cls_scores, query_labels)
        return losses

    def forward_support(self, x, gt_label, **kwargs):
        """Forward support data in meta testing."""
        self.support_feats.append(x)
        self.support_labels.append(gt_label)

    def forward_query(self, x, **kwargs):
        """Forward query data in meta testing."""
        assert self.prototype_feats is not None
        cls_scores = -1 * torch.cdist(
            x.unsqueeze(0), self.prototype_feats.unsqueeze(0)).squeeze(0)
        pred = F.softmax(cls_scores, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # reset prototype features for testing new task
        self.support_feats.clear()
        self.support_labels.clear()
        self.prototype_feats = None

    def before_forward_query(self):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        feats = torch.cat(self.support_feats, dim=0)
        labels = torch.cat(self.support_labels, dim=0)
        class_ids, _ = torch.unique(labels).sort()
        prototype_feats = [
            feats[labels == class_id].mean(0, keepdim=True)
            for class_id in class_ids
        ]
        self.prototype_feats = torch.cat(prototype_feats, dim=0)
