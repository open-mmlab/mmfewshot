# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=50)
evaluation = dict(by_epoch=True, metric='accuracy', interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
pin_memory = True
use_infinite_sampler = True
seed = 0
