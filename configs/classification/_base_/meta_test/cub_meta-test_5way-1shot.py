img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(int(img_size * 1.15), -1)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

num_ways = 5
num_shots = 1
num_queries = 15
num_val_episodes = 100
num_test_episodes = 2000

data = dict(
    val=dict(
        type='MetaTestDataset',
        num_episodes=num_val_episodes,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        dataset=dict(
            type='CUBDataset',
            subset='val',
            data_prefix='data/CUB_200_2011',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=num_val_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=False,
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization is a time consuming operation
            support=dict(batch_size=num_ways * num_shots, num_workers=0),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))),
    test=dict(
        type='MetaTestDataset',
        num_episodes=num_test_episodes,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        # seed for generating meta test episodes
        episodes_seed=0,
        dataset=dict(
            type='CUBDataset',
            subset='test',
            data_prefix='data/CUB_200_2011',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=num_test_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=False,
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization for each task is a time consuming operation
            support=dict(batch_size=num_ways * num_shots, num_workers=0),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))))
