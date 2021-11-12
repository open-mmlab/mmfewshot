# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.builder import CLASSIFIERS

from .base_finetune import BaseFinetuneClassifier
from .base_metric import BaseMetricClassifier
from .baseline import Baseline
from .baseline_plus import BaselinePlus
from .maml import MAML
from .matching_net import MatchingNet
from .meta_baseline import MetaBaseline
from .neg_margin import NegMargin
from .proto_net import ProtoNet
from .relation_net import RelationNet

__all__ = [
    'CLASSIFIERS', 'BaseFinetuneClassifier', 'BaseMetricClassifier',
    'Baseline', 'BaselinePlus', 'ProtoNet', 'MatchingNet', 'RelationNet',
    'NegMargin', 'MetaBaseline', 'MAML'
]
