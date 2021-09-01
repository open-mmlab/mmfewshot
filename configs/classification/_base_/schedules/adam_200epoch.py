runner = dict(type='EpochBasedRunner', max_epochs=200)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed', warmup=None)
