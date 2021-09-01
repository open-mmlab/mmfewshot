from mmcls.models.builder import CLASSIFIERS

from .baseline import BaselineClassifier
from .baseline_pp import BaselinePPClassifier
from .finetune_base import FinetuneBaseClassifier
from .maml import MAMLClassifier
from .matching_net import MatchingNetClassifier
from .meta_baseline import MetaBaselineClassifier
from .meta_metric_base import MetaMetricBaseClassifier
from .neg_margin import NegMarginClassifier
from .proto_net import ProtoNetClassifier
from .relation_net import RelationNetClassifier

__all__ = [
    'CLASSIFIERS', 'FinetuneBaseClassifier', 'MetaMetricBaseClassifier',
    'BaselineClassifier', 'BaselinePPClassifier', 'ProtoNetClassifier',
    'MatchingNetClassifier', 'RelationNetClassifier', 'NegMarginClassifier',
    'MetaBaselineClassifier', 'MAMLClassifier'
]
