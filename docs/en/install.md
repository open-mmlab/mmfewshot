## Prerequisites

- Linux (Windows is not officially supported)
- Python 3.7+
- PyTorch 1.5+
- CUDA 9.2+
- GCC 5+
- [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 1.3.12+
- [mmdet](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) 2.16.0+
- [mmcls](https://mmclassification.readthedocs.io/en/latest/install.html) 0.15.0+


Compatible MMCV, MMClassification and MMDetection versions are shown as below. Please install the correct version of them to avoid installation issues.

| MMFewShot version   |    MMCV version   |      MMClassification version     |      MMDetection version     |
|:-------------------:|:-----------------:|:---------------------------------:|:----------------------------:|
| master              | mmcv-full>=1.3.12 |      mmdet >= 2.16.0              |      mmcls >=0.15.0          |

**Note:** You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.7, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
    ```


### Install MMFewShot

It is recommended to install MMFewShot with [MIM](https://github.com/open-mmlab/mim),
which automatically handle the dependencies of OpenMMLab projects, including mmcv and other python packages.

```shell
pip install openmim
mim install mmfewshot
```

Or you can still install MMFewShot manually:

1. Install mmcv-full.

    ```shell
    # pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
    ```

    mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

    ```
    # We can ignore the micro version of PyTorch
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

    Optionally you can compile mmcv from source if you need to develop both mmcv and mmfewshot. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

2. Install MMClassification and MMDetection.

    You can simply install mmclassification and mmdetection with the following command:

    ```shell
    pip install mmcls mmdet
    ```

3. Install MMFewShot.

    You can simply install mmfewshot with the following command:

    ```shell
    pip install mmfewshot
    ```

    or clone the repository and then install it:

    ```shell
    git clone https://github.com/open-mmlab/mmfewshot.git
    cd mmfewshot
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"

**Note:**

a. When specifying `-e` or `develop`, MMFewShot is installed on dev mode
, any local modifications made to the code will take effect without reinstallation.

b. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will
 only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.


### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmfewshot/blob/master/docker/Dockerfile) to build an image. Ensure that you are using [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmfewshot docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmfewshot/data mmfewshot
```

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMDetection with conda.

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# install mmclassification mmdetection
pip install mmcls mmdet

# install mmfewshot
git clone https://github.com/open-mmlab/mmfewshot.git
cd mmfewshot
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```


## Verification

To verify whether MMFewShot is installed correctly, we can run the demo code and inference a demo image.

Please refer to [few shot classification demo](https://github.com/open-mmlab/mmfewshot/tree/main/demo#few-shot-classification-demo)
or [few shot detection demo](https://github.com/open-mmlab/mmfewshot/tree/main/demo#few-shot-detection-demo)
 for more details. The demo code is supposed to run successfully upon you finish the installation.

## Dataset Preparation
Please refer to [data preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data) for dataset preparation.
