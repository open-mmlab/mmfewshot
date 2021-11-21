# Few Shot Detection Data Preparation

<!-- [DATASET] -->

```bibtex
@article{everingham2010pascal,
    title={The pascal visual object classes (voc) challenge},
    author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
    journal={International journal of computer vision},
    volume={88},
    number={2},
    pages={303--338},
    year={2010},
    publisher={Springer}
}

@inproceedings{kang2019few,
    title={Few-shot Object Detection via Feature Reweighting},
    author={Kang, Bingyi and Liu, Zhuang and Wang, Xin and Yu, Fisher and Feng, Jiashi and Darrell, Trevor},
    booktitle={ICCV},
    year={2019}
}
```

## download VOC dataset
The VOC 2007/2012 dataset can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/).

In mmfewshot, VOC 2007(trainval) + 2012 (trainval) are used for training and VOC 2007(test) is used for evaluation.

The data structure is as follows:
```none
mmfewshot
├── mmfewshot
├── tools
├── configs
├── data
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

## download few shot annotations
In mmfewshot, we use the VOC few shot split released in TFA [repo](https://github.com/ucbdrive/few-shot-object-detection).
The original data spilt can be found in [here](http://dl.yf.io/fs-det/datasets/vocsplit/).

We provide a re-organized data split.
Please download [voc.tar.gz](https://download.openmmlab.com/mmfewshot/few_shot_ann/voc.tar.gz)
and unzip them into `$MMFEWSHOT/data/few_shot_ann`.



The final data structure is as follows:
```none
mmfewshot
├── mmfewshot
├── tools
├── configs
├── data
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
│   ├── few_shot_ann
│   │   ├── voc
│   │   │   ├── benchmark_1shot
│   │   │   ├── benchmark_2shot
│   │   │   ├── benchmark_3shot
│   │   │   ├── benchmark_5shot
│   │   │   ├── benchmark_10shot
```
