# Learn the Basics

This chapter introduces you to the basic conception of few shot learning, and the framework of
MMFewShot, and provides links to detailed tutorials about MMFewShot.

## What is Few Shot Learning

### Problem definition
Few-shot learning aims at generalizing to new tasks based on a limited number of samples using prior knowledge.
The prior knowledge usually refers to a large scale training set which has many classes and samples,
while the samples in new tasks are never seen in the training set.
For example, in few shot image classification, a pre-trained model only can see
five bird images (each class has one image and doesn't exist in the pretrained dataset) and
predict the bird in query image belongs to which class.
And in few shot detection, a detector needs to detect the new categories based on a few instances.

In summary, few shot learning focus on two aspects:
- how to embed prior knowledge into models (pre-train with large scale dataset)
- how to transfer knowledge to adapt new tasks (learn on a few labeled samples).


<div align=center>
<img src="https://github.com/open-mmlab/mmfewshot/blob/main/resources/demo.png?raw=true" width=80%/>
</div>

### Terminologies in few-shot learning
- Training set: every class in the training set has many samples, and it is big enough for learning a deep neural network.
- Support set: a small set of labeled images and all the classes do not exist in training set.
- Query set: unlabeled images to predict and share the same classes with support set.

- N way K shot: the support set setting
   - N-way means the support set has k classes.
   - K-shot means every class has n samples.


### Evaluation
#### Few shot classification
Meta test: randomly sample different tasks (N way K shot + Q query images)
and calculate the accuracy mean and std of the prediction on query images.
#### Few shot detection
The novel class performance (mAP or AP50) on test set.



### The basic pipeline for few shot learning
We will introduce a simple baseline for all the few shot learning task to further illustrate how few shot learning work.
The most obvious pipeline is fine-tuning.
It usually consists of two steps: train a model on a large scale dataset and then fine-tune on few shot data.
For image classification, we first pretrain a model with training set using cross-entropy loss, and then
we can transfer the backbone and fine tune a new classification head.
For detection, we can first pretrain a faster-rcnn on training set, and
then fine tune a new bbox head on a few instances to detect novel class.
In many cases, the fine-tuning is a simple but effective strategy for few shot learning.

## What is MMFewShot

MMFewShot is the first toolbox that provides a framework for unified implementation and evaluation of few shot classification and detection methods,
and below is its whole framework:

<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/143182168-e2b0a3b8-4dce-4e44-b134-a2577c8290c5.png" width=80%/>
</div>

MMFewShot consists of 4 main parts, `datasets`, `models`, `core` and `apis`.

- `datasets` is for datasets loading and data augmentation. In this part,
we support various datasets for classification and detection algorithms,
useful data augmentation transforms in `pipelines` for pre-processing image
and  `datasetswrappers` for data sampling.

- `models` is the most vital part containing models and loss functions.

- `core` provides evaluation tools and customized hooks for model training and evaluation.

- `apis` provides high-level APIs for models training, testing, and inference,

## How to Use this Guide

Here is a detailed step-by-step guide to learn more about MMFewShot:

1. For installation instructions, please see [install](install.md).

2. [get_started](get_started.md) is for the basic usage of MMFewShot.

3. Refer to the below tutorials to dive deeper:

- Few Shot Classification
    - [Overview](classification/overview.md)
    - [Config](classification/customize_config.md)
    - [Customize Dataset](classification/customize_dataset.md)
    - [Customize Model](classification/customize_models.md)
    - [Customize Runtime](classification/customize_runtime.md)

- Few Shot Detection
    - [Overview](detection/overview.md)
    - [Config](detection/customize_config.md)
    - [Customize Dataset](detection/customize_dataset.md)
    - [Customize Model](detection/customize_models.md)
    - [Customize Runtime](detection/customize_runtime.md)
