img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromBytes'),
    dict(type='Resize', size=(int(img_size * 1.15), -1)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

num_ways = 5
num_shots = 5
num_queries = 15

data = dict(
    val=dict(
        type='MetaTestDataset',
        dataset=dict(
            type='TieredImageNetDataset',
            subset='val',
            data_prefix='data/tiered_imagenet',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=100,
            num_ways=num_ways,
            num_shots=num_shots,
            num_queries=num_queries,
            # cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            # initialize worker is a time consuming operation
            support=dict(batch_size=num_ways * num_shots, num_workers=0),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))),
    test=dict(
        type='MetaTestDataset',
        dataset=dict(
            type='TieredImageNetDataset',
            subset='test',
            data_prefix='data/tiered_imagenet',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=1000,
            num_ways=num_ways,
            num_shots=num_shots,
            num_queries=num_queries,
            # cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            # initialize worker is a time consuming operation
            support=dict(batch_size=num_ways * num_shots, num_workers=0),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))))
