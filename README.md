[comment]: <> (<div align="center">)

[comment]: <> (  <img src="resources/mmfewshot-logo.png" width="500px"/>)

[comment]: <> (</div>)

## Introduction

[comment]: <> (English | [简体中文]&#40;/README_zh-CN.md&#41;)

[comment]: <> ([![Documentation]&#40;https://readthedocs.org/projects/mmfewshot/badge/?version=latest&#41;]&#40;https://mmfewshot.readthedocs.io/en/latest/?badge=latest&#41;)

[comment]: <> ([![actions]&#40;https://github.com/open-mmlab/mmfewshot/workflows/build/badge.svg&#41;]&#40;https://github.com/open-mmlab/mmfewshot/actions&#41;)

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/open-mmlab/mmfewshot/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/open-mmlab/mmfewshot&#41;)

[comment]: <> ([![PyPI]&#40;https://badge.fury.io/py/mmedit.svg&#41;]&#40;https://pypi.org/project/mmedit/&#41;)

[comment]: <> ([![LICENSE]&#40;https://img.shields.io/github/license/open-mmlab/mmfewshot.svg&#41;]&#40;https://github.com/open-mmlab/mmfewshot/blob/master/LICENSE&#41;)

[comment]: <> ([![Average time to resolve an issue]&#40;https://isitmaintained.com/badge/resolution/open-mmlab/mmfewshot.svg&#41;]&#40;https://github.com/open-mmlab/mmfewshot/issues&#41;)

[comment]: <> ([![Percentage of issues still open]&#40;https://isitmaintained.com/badge/open/open-mmlab/mmfewshot.svg&#41;]&#40;https://github.com/open-mmlab/mmfewshot/issues&#41;)


mmfewshot is an open source few shot learning toolbox based on PyTorch. It is a part of the [OpenMMLab](https://open-mmlab.github.io/) project.

The master branch works with **PyTorch 1.7+**.
The compatibility to earlier versions of PyTorch is not fully tested.

Documentation: https://mmfewshot.readthedocs.io/en/latest/.


### Major features


## Model Zoo

Supported algorithms:

<details open>
<summary>classification</summary>

- [x] [Baseline](configs/classification/baseline/README.md) (ICLR'2019)
- [x] [Baseline++](configs/classification/baseline_plus/README.md) (ICLR'2019)
- [x] [NegMargin](configs/classification/neg_margin/README.md) (ECCV'2020)
- [x] [MatchingNet](configs/classification/matching_net/README.md) (NeurIPS'2016)
- [x] [ProtoNet](configs/classification/proto_net/README.md) (NeurIPS'2017)
- [x] [RelationNet](configs/classification/relation_net/README.md) (CVPR'2018)
- [x] [MetaBaseline](configs/classification/meta_baseline/README.md) (ICCV'2021)
- [x] [MAML](configs/classification/maml/README.md) (ICML'2017)

</details>

<details open>
<summary>Detection</summary>

- [x] [TFA](configs/detection/tfa/README.md) (ICML'2020)
- [x] [FSCE](configs/detection/fsce/README.md) (CVPR'2021)
- [x] [AttentionRPN](configs/detection/attention_rpn/README.md) (CVPR'2020)
- [x] [MetaRCNN](configs/detection/meta_rcnn/README.md) (ICCV'2019)
- [x] [FSDetView](configs/detection/fsdetview/README.md) (ECCV'2020)
- [x] [MPSR](configs/detection/mpsr/README.md) (ECCV'2020)

</details>


Please refer to [model_zoo](https://mmfewshot.readthedocs.io/en/latest/modelzoo.html) for more details.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog


## Installation

Please refer to [install.md](docs/get_started.md) for installation.

## Get Started

Please see [getting_started.md](docs/get_started.md) for the basic usage of mmfewshot.



## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmfewshot2020,
    title={OpenMMLab Few Shot Learning Toolbox and Benchmark},
    author={mmfewshot Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmfewshot}},
    year={2021}
}
```


## Contributing

We appreciate all contributions to improve mmfewshot. Please refer to [CONTRIBUTING.md in MMFewShot](https://github.com/open-mmlab/mmcv/blob/master/.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

mmfewshot is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmfewshot): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): A powerful toolkit for generative models.
