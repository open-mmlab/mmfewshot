import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS

from mmfewshot.classification.datasets import label_wrapper
from .base_head import FewShotBaseHead


@HEADS.register_module()
class MatchingHead(FewShotBaseHead):
    """Classification head for `MatchingNet.

    <https://arxiv.org/abs/1606.04080>`_.

    Note that this implementation is without FCE(Full Context Embeddings).

    Args:
        temperature (float): The scale factor of `cls_score`.
        loss (dict): Config of training loss.
    """

    def __init__(self,
                 temperature=100,
                 loss=dict(type='NLLLoss', loss_weight=1.0),
                 *args,
                 **kwargs):
        super(MatchingHead, self).__init__(loss=loss, *args, **kwargs)
        self.temperature = temperature
        self.support_feats = []
        self.support_labels = []
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
        cosine_distance = torch.mm(
            F.normalize(query_feats),
            F.normalize(support_feats).transpose(0, 1))
        scores = F.softmax(cosine_distance * self.temperature, dim=-1)
        scores = torch.cat([
            scores[:, support_labels == class_id].mean(1, keepdim=True)
            for class_id in class_ids
        ],
                           dim=1).log()
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
            F.normalize(self.support_feats).transpose(0, 1))
        scores = F.softmax(cosine_distance * self.temperature, dim=-1)
        scores = torch.cat([
            scores[:, self.support_labels == class_id].mean(1, keepdim=True)
            for class_id in self.class_ids
        ],
                           dim=1)
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

    def before_forward_query(self):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.support_feats = torch.cat(self.support_feats, dim=0)
        self.support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids, _ = torch.unique(self.support_labels).sort()
