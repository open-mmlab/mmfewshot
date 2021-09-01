runner = dict(type='IterBasedRunner', max_iters=100000)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed', warmup=None)
