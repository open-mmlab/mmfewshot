def check_config(cfg):
    """Check for missing or deprecated arguments."""
    support_tasks = ['mmcls', 'mmdet']
    if 'task_type' not in cfg:
        raise AttributeError(f'Please set `task_type` '
                             f'in your config, {support_tasks} are supported')
    if cfg.task_type not in support_tasks:
        raise ValueError(f'{support_tasks} are supported, '
                         f'but get `task_type` {cfg.task_type}')
    return cfg
