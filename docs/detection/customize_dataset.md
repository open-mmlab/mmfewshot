# Tutorial 2: Adding New Dataset


## Customize Dataset

### Load annotations from file
Different from the config in mmdet using `ann_file` to load a single dataset, we use `ann_cfg` to support the complex few shot setting.

The `ann_cfg` is a list of dict and support two type of file:
- loading annotation from the regular `ann_file` of dataset.
    ```python
    ann_cfg = [dict(type='ann_file', ann_file='path/to/ann_file'), ...]
    ```
    For `FewShotVOCDataset`, we also support load specific class from `ann_file` in `ann_classes`.
    ```python
    dict(type='ann_file', ann_file='path/to/ann_file', ann_classes=['dog', 'cat'])
    ```

- loading annotation from a json file saved by a dataset.
    ```python
    ann_cfg = [dict(type='saved_dataset', ann_file='path/to/ann_file'), ...]
    ```
    To save a dataset, we can set the `save_dataset=True` in config file,
    and the dataset will be saved as `${WORK_DIR}/{TIMESTAMP}_saved_data.json`
    ```python
    dataset=dict(type='FewShotVOCDataset', save_dataset=True, ...)
    ```

### Load annotations from predefined benchmark

Unlike few shot classification can test on thousands of tasks in a short time,
it is hard to follow the same protocol in few shot detection because of the computation cost.
Thus, we provide the predefined data split for reproducibility.
These data splits directly use the files released from TFA [repo](https://github.com/ucbdrive/few-shot-object-detection).
The details of data preparation can refer to [here](https://github.com/open-mmlab/mmfewshot/tree/master/tools/data/detection).

To load these predefined data splits, the type of dataset need to be set to
`FewShotVOCDefaultDataset` or `FewShotCocoDefaultDataset`.
We provide data splits of each reproduced checkpoint for each method.
In config file, we can use `method` and `setting` to determine which data split to load.

Here is an example of config:
```python
dataset = dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_1SHOT')]
)
```

### Load annotations from another dataset during runtime

In few shot setting, we can use `FewShotVOCCopyDataset` or `FewShotCocoCopyDataset` to copy a dataset from other dataset
during runtime for some special cases, such as copying online random sampled support set for model initialization before evaluation.
It needs user to modify code in `mmfewshot.detection.apis`.
More details can refer to mmfewshot/detection/apis/train.py.
Here is an example of config:
```python
dataset = dict(
        type='FewShotVOCCopyDataset',
        ann_cfg=[dict(data_infos=FewShotVOCDataset.data_infos)])
```

### Use predefined class splits
The predefined class splits are supported in datasets.
For VOC, we support [`ALL_CLASSES_SPLIT1`,`ALL_CLASSES_SPLIT2`, `ALL_CLASSES_SPLIT3`,
`NOVEL_CLASSES_SPLIT1`, `NOVEL_CLASSES_SPLIT2`, `NOVEL_CLASSES_SPLIT3`, `BASE_CLASSES_SPLIT1`,
`BASE_CLASSES_SPLIT2`, `BASE_CLASSES_SPLIT3`].
For COCO, we support [`ALL_CLASSES`, `NOVEL_CLASSES`, `BASE_CLASSES`]

Here is an example of config:
```python
data = dict(
    train=dict(type='FewShotVOCDataset', classes='ALL_CLASSES_SPLIT1'),
    val=dict(type='FewShotVOCDataset', classes='ALL_CLASSES_SPLIT1'),
    test=dict(type='FewShotVOCDataset', classes='ALL_CLASSES_SPLIT1'))
```

Also, the class splits can be used to report the evaluation results on different class splits.
Here is an example of config:
```python
evaluation = dict(class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
```

### Customize the number of annotations
For FewShotDataset, we support two ways to filter extra annotations.
- `ann_shot_filter`: use a dict to specify the class, and
  the corresponding maximum number of instances when loading
  the annotation file.
  For example, we only want 10 instances of dog and 5 instances of person, while other
  instances from other classes remain unchanged:
  ```python
  dataset=dict(type='FewShotVOCDataset',
               ann_shot_filter=dict(dog=10, person=5),
               ...)
  ```
- `num_novel_shots` and `num_base_shots`: use predefined class splits
  to indicate the corresponding maximum number of instances.
  For example, we only want 1 instance for each novel class and 3 instances for base class:
  ```python
  dataset=dict(
      type='FewShotVOCDataset',
      num_novel_shots=1,
      num_base_shots=2,
      ...)
  ```

### Customize the organization of annotations
We also support to split the annotation into instance wise, i.e. each image only have one instance,
and the images can be repeated.
```python
dataset=dict(
    type='FewShotVOCDataset',
    instance_wise=True,
    ...)
```

### Customize pipeline
To support different pipelines in single dataset, we can use `multi_pipelines`.
In config file, `multi_pipelines` use the name of keys to indicate specific piplines.
Here is an example of config:
```python
multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
train=dict(
    type='NWayKShotDataset',
    dataset=dict(
        type='FewShotCocoDataset',
        ...
        multi_pipelines=train_multi_pipelines))
```
When `multi_pipelines` is used, we need to specific the pipeline names in
`prepare_train_img` to fetch the image.
For example
```python
dataset.prepare_train_img(self, idx, 'query')
```


## Customize Dataset Wrapper
In few shot setting, the various sampling logic is implemented by
dataset wrapper.
An example of customizing query-support data sampling logic for training:

#### Create a new dataset wrapper
We can create a new dataset wrapper in mmfewshot/detection/datasets/dataset_wrappers.py to customize sampling logic.

```python
class MyDatasetWrapper:
    def __init__(self, dataset, support_dataset=None, args_a, args_b, ...):
        # query_dataset and support_dataset can use same dataset
        self.query_dataset = dataset
        self.support_dataset = support_dataset
        if support_dataset is None:
            self.support_dataset = dataset
        ...

    def __getitem__(self, idx):
        ...
        query_data = self.query_dataset.prepare_train_img(idx, 'query')
        # customize sampling logic
        support_idxes = ...
        support_data = [
            self.support_dataset.prepare_train_img(idx, 'support')
            for idx in support_idxes
        ]
        return {'query_data': query_data, 'support_data': support_data}

```

#### Update dataset builder
We need to add the building code in mmfewshot/detection/datasets/builder.py
for our customize dataset wrapper.


```python
def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    ...
    elif cfg['type'] == 'MyDatasetWrapper':
        dataset = MyDatasetWrapper(
            build_dataset(cfg['dataset'], default_args),
            build_dataset(cfg['support_dataset'], default_args) if cfg.get('support_dataset', False) else None,
            # pass customize arguments
            args_a=cfg['args_a'],
            args_b=cfg['args_b'],
            ...)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
```

#### Update dataloader builder
We need to add the building code of dataloader in mmfewshot/detection/datasets/builder.py,
when the customize dataset wrapper will return list of Tensor.
We can use `multi_pipeline_collate_fn` to handle this case.


```python
def build_dataset(cfg, default_args=None):
    ...
    if isinstance(dataset, MyDatasetWrapper):
      from mmfewshot.utils import multi_pipeline_collate_fn
      # `multi_pipeline_collate_fn` are designed to handle
      # the data with list[list[DataContainer]]
      data_loader = DataLoader(
          dataset,
          batch_size=batch_size,
          sampler=sampler,
          num_workers=num_workers,
          collate_fn=partial(
              multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
          pin_memory=False,
          worker_init_fn=init_fn,
          **kwargs)
    ...
```


#### Update the arguments in model

The argument names in forward function need to be consistent with the customize dataset wrapper.

```python
class MyDetector(BaseDetector):
    ...
    def forward(self, query_data, support_data, ...):
        ...
```

#### using customize dataset wrapper in config
Then in the config, to use `MyDatasetWrapper` you can modify the config as the following,
```python
dataset_A_train = dict(
        type='MyDatasetWrapper',
        args_a=None,
        args_b=None,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            multi_pipelines=train_multi_pipelines
        ),
        support_dataset=None
    )
```


## Customize Dataloader Wrapper
We also support to iterate two different dataset simultaneously by dataloader wrapper.

An example of customizing dataloader wrapper for query and support dataset:
#### Create a new dataloader wrapper
We can create a new dataset wrapper in mmfewshot/detection/datasets/dataloader_wrappers.py to customize sampling logic.

```python
class MyDataloader:
    def __init__(self, query_data_loader, support_data_loader):
        self.dataset = query_data_loader.dataset
        self.sampler = query_data_loader.sampler
        self.query_data_loader = query_data_loader
        self.support_data_loader = support_data_loader

    def __iter__(self):
        self.query_iter = iter(self.query_data_loader)
        self.support_iter = iter(self.support_data_loader)
        return self

    def __next__(self):
        query_data = self.query_iter.next()
        support_data = self.support_iter.next()
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self) -> int:
        return len(self.query_data_loader)
```

#### Update dataloader builder
We need to add the build code in mmfewshot/detection/datasets/builder.py
for our customize dataset wrapper.


```python
def build_dataloader(dataset, ...):
    if isinstance(dataset, MyDataset):
        ...
        query_data_loader = DataLoader(...)
        support_data_loader = DataLoader(...)
        # wrap two dataloaders with dataloader wrapper
        data_loader = MyDataloader(
            query_data_loader=query_data_loader,
            support_data_loader=support_data_loader)

    return dataset
```
