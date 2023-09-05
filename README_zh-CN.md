<div align="center">
  <img src="resources/mmfewshot-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmfewshot)](https://pypi.org/project/mmfewshot/)
[![PyPI](https://img.shields.io/pypi/v/mmfewshot)](https://pypi.org/project/mmfewshot)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmfewshot.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmfewshot/workflows/build/badge.svg)](https://github.com/open-mmlab/mmfewshot/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmfewshot/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmfewshot)
[![license](https://img.shields.io/github/license/open-mmlab/mmfewshot.svg)](https://github.com/open-mmlab/mmfewshot/blob/master/LICENSE)

[📘Documentation](https://mmfewshot.readthedocs.io/zh_CN/latest/) |
[🛠️Installation](https://mmfewshot.readthedocs.io/zh_CN/latest/install.html) |
[👀Model Zoo](https://mmfewshot.readthedocs.io/zh_CN/latest/model_zoo.html) |
[🆕Update News](https://mmfewshot.readthedocs.io/zh_CN/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmfewshot/issues/new/choose)

</div>

<div align="center">

[English](/README.md) | 简体中文

</div>

## Introduction

MMFewShot 是一款基于 PyTorch 的少样本学习代码库，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

<div align="left">
  <img src="resources/demo.png"/>
</div>

### 主要特性

- **支持多种少样本任务**

  MMFewShot 为少样本分类和检测任务提供了的统一实现和评估框架。

- **模块化设计**

  MMFewShot 将不同少样本任务解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的少样本算法模型。

- **强大的基准模型与SOTA**

  MMFewShot 提供了少样本分类和检测任务中最先进的算法和强大的基准模型.

## 更新

v0.1.0 版本已于 2021 年 11 月 24 日发布，可通过查阅[更新日志](docs/zh_cn/changelog.md)了解更多细节以及发布历史。

## 安装与准备数据集

MMFewShot 依赖 [PyTorch](https://pytorch.org/) 和 [MMCV](https://github.com/open-mmlab/mmcv) 。
请参考[安装文档](docs/zh_cn/install.md)进行安装和参考[数据准备](tools/data/README.md)准备数据集。

## 开始使用 MMFewShot

如果初次了解少样本学习，你可以从[基础介绍](docs/en/intro.md)开始了解少样本学习的基本概念和 MMFewShot 的框架。
如果对少样本学习很熟悉，请参考[使用教程](docs/zh_cn/get_started.md)获取MMFewShot的基本用法。

MMFewShot 也提供了其他更详细的教程，包括：

- 少样本分类

  - [概览](docs/zh_cn/classification/overview.md)
  - [配置文件](docs/zh_cn/classification/customize_config.md)
  - [添加数据集](docs/zh_cn/classification/customize_dataset.md)
  - [添加新模型](docs/zh_cn/classification/customize_models.md)
  - [自定义模型运行环境](docs/zh_cn/classification/customize_runtime.md)。

- 少样本检测

  - [概览](docs/zh_cn/detection/overview.md)
  - [配置文件](docs/zh_cn/detection/customize_config.md)
  - [添加数据集](docs/zh_cn/detection/customize_dataset.md)
  - [添加新模型](docs/zh_cn/detection/customize_models.md)
  - [自定义模型运行环境](docs/zh_cn/detection/customize_runtime.md)。

## 基准测试和模型库

本工具箱支持的各个模型的结果和设置都可以在[模型库](docs/en/model_zoo.md)页面中查看。

已支持的算法：

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

## 参与贡献

我们感谢所有的贡献者为改进和提升 MMFewShot 所作出的努力。请参考[贡献指南](https://github.com/open-mmlab/mmfewshot/blob/main/.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMFewShot 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。

我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```bibtex
@misc{mmfewshot2021,
    title={OpenMMLab Few Shot Learning Toolbox and Benchmark},
    author={mmfewshot Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmfewshot}},
    year={2021}
}
```

## 许可

该项目遵循[Apache 2.0 license](/LICENSE)开源协议。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，扫描下方微信二维码添加喵喵好友，进入 OpenMMLab 微信交流社群。【加好友申请格式：研究方向+地区+学校/公司+姓名】

<div align="center">
<img src="https://user-images.githubusercontent.com/24663779/116371114-a8005e80-a83d-11eb-9123-17fc9cfe7475.jpg" height="400" />  <img src="https://github.com/open-mmlab/mmfewshot/assets/62195058/ebb804d4-288c-4522-a4d3-8ff7e54d56ef" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
