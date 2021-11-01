# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.builder import HEADS

from .cosine_distance_head import CosineDistanceHead
from .linear_head import LinearHead
from .matching_head import MatchingHead
from .meta_baseline_head import MetaBaselineHead
from .neg_margin_head import NegMarginHead
from .prototype_head import PrototypeHead
from .relation_head import RelationHead

__all__ = [
    'HEADS', 'MetaBaselineHead', 'MatchingHead', 'NegMarginHead', 'LinearHead',
    'CosineDistanceHead', 'PrototypeHead', 'RelationHead'
]
