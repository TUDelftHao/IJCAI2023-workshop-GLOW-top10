


## This repository is the implemented based on "Large Selective Kernel Network for Remote Sensing Object Detection" at: [https://arxiv.org/pdf/2303.09030.pdf](https://arxiv.org/pdf/2303.09030.pdf), the original repo is here: https://github.com/zcablii/Large-Selective-Kernel-Network.


## Introduction

Base on existing amazing work, some new features and toolkits used for oriented obeject detection task are implemented. 


## Results and models

Imagenet 300-epoch pre-trained LSKNet-T backbone: [Download](https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_t_backbone-2ef8a593.pth)

Imagenet 300-epoch pre-trained LSKNet-S backbone: [Download](https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_s_backbone-e9d2e551.pth)




## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/zcablii/Large-Selective-Kernel-Network.git
cd Large-Selective-Kernel-Network
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)





## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{li2023large,
  title   = {Large Selective Kernel Network for Remote Sensing Object Detection},
  author  = {Li, Yuxuan and Hou, Qibin and Zheng, Zhaohui and Cheng, Mingming and Yang, Jian and Li, Xiang},
  journal={ArXiv},
  year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
