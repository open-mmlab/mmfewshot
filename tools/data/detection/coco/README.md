# Preparing COCO Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{lin2014microsoft,
    title={Microsoft coco: Common objects in context},
    author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
    booktitle={European conference on computer vision},
    pages={740--755},
    year={2014},
    organization={Springer}
}

@inproceedings{kang2019few,
    title={Few-shot Object Detection via Feature Reweighting},
    author={Kang, Bingyi and Liu, Zhuang and Wang, Xin and Yu, Fisher and Feng, Jiashi and Darrell, Trevor},
    booktitle={ICCV},
    year={2019}
}
```

## download coco dataset
The coco14/coco17 dataset can be downloaded from [here](https://cocodataset.org/#download).

In mmfewshot, coco14 is used as default setting, while coco17 is optional.
Some methods (attention rpn) were proposed with coco17 data split, which is also evaluated in mmfewshot.

The data structure is as follows:
```none
mmfewshot
├── mmfewshot
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2014
│   │   ├── val2014
│   │   ├── train2017 (optional)
│   │   ├── val2017 (optional)
```

## download few shot annotations
In mmfewshot, we use the train/val/few shot split of coco14 released in TFA [repo](https://github.com/ucbdrive/few-shot-object-detection).
The original data spilt can be found in [here](http://dl.yf.io/fs-det/datasets/cocosplit/).

We provide a re-organized data split.
Please download [coco.tar.gz](https://download.openmmlab.com/mmfewshot/few_shot_ann/coco.tar.gz)
and unzip them into `$MMFEWSHOT/data/few_shot_ann`.

The final data structure is as follows:
```none
mmfewshot
├── mmfewshot
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2014
│   │   ├── val2014
│   │   ├── train2017 (optional)
│   │   ├── val2017 (optional)
│   ├── few_shot_ann
│   │   ├── coco
│   │   │   ├── annotations
│   │   │   │   ├── train.json
│   │   │   │   ├── val.json
│   │   │   ├── attention_rpn_10shot (for coco17)
│   │   │   ├── benchmark_10shot
│   │   │   ├── benchmark_30shot
```
