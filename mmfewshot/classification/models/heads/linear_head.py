import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS

from .base_head import FewShotBaseHead


@HEADS.register_module()
class LinearHead(FewShotBaseHead):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes, in_channels, *args, **kwargs):
        super(LinearHead, self).__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} ' \
                                f'must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.init_layers()

    def init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward_train(self, x, gt_label, **kwargs):
        """Forward training data."""
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

    def forward_support(self, x, gt_label, **kwargs):
        """Forward support data in meta testing."""
        return self.forward_train(x, gt_label, **kwargs)

    def forward_query(self, x, **kwargs):
        """Forward query data in meta testing."""
        cls_score = self.fc(x)
        pred = F.softmax(cls_score, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        self.init_layers()
        self.train()

    def before_forward_query(self):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.eval()
