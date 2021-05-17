import pytest
from mmcv import ConfigDict

from mmfewshot.utils.check_config import check_config


def test_check_config():
    config = dict(task_type='mmdet')
    cfg = ConfigDict(config)
    check_config(cfg)
    with pytest.raises(AttributeError):
        cfg.pop('task_type')
        check_config(cfg)
    with pytest.raises(ValueError):
        cfg.task_type = 'cls'
        check_config(cfg)
