# JointFontGAN in PyTorch

This is the implementation of the [JointFontGAN: Joint Geometry-Content GAN for Font Generation via Few-Shot Learning](https://dl.acm.org/doi/10.1145/3394171.3413705). The code was written by [Yankun Xi](https://github.com/yankunxi).
If you use this code for your research, please cite:


## Prerequisites:
- Linux or macOS
- Python 3.6 or later (latest built on Python 3.8)
- Pytorch 1.2 or later (latest built on Pytorch 1.6)
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
```bash
pip install visdom
pip install dominate
pip install scikit-image
```

- Clone this repo:
```bash
mkdir JointFontGAN
cd JointFontGAN
git clone https://github.com/never-witness/xifontgan
mkdir dataset
```

- Download dataset:

Download the following two datasets into `dataset` folder and unzip.

Capitals64 dataset: https://drive.google.com/file/d/1qrxhhgG2vwUhhq-shbHxzt3b1ahkoNt_/view?usp=sharing

SandunLK10k64 dataset: https://drive.google.com/file/d/1VgzxiBrYYUdB0eyNKVb137W0jY43YCeM/view?usp=sharing

- Enter this repo:
```bash
cd xifontgan
```

- (Optional) Download pre-trained model

Download the following models into `checkpoints` folder and unzip.

Capitals54 dataset: https://drive.google.com/file/d/1C3JvbjdRecqVc3UmWxR1mLP_i0hwmDxp/view?usp=sharing

SandunLK10k64 dataset: https://drive.google.com/file/d/1T140Uig4CfL8W6vsh0TElkguBAJrf_Rp/view?usp=sharing



### JointFontGAN train/test

- To train the model, please run the following scripts for the two datasets:

```bash
. ./scripts/EskGAN/XItrain_EskGAN.sh Capitals64
```

```bash
. ./scripts/EskGAN/XItrain_EskGAN2_dspostG=1.sh SandunLK10k64
```

Or you can skip the training phase and test on our pretrained models.

- To test the model:

```bash
. ./scripts/EskGAN/XItest_EskGAN.sh Capitals64 test
```

```bash
. ./scripts/EskGAN/XItest_EskGAN2_dspostG=1.sh SandunLK10k64 test
```

### Citation

```
@inproceedings{xi2020jointfontgan,
  title={JointFontGAN: Joint Geometry-Content GAN for Font Generation via Few-Shot Learning},
  author={Xi, Yankun and Yan, Guoli and Hua, Jing and Zhong, Zichun},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={4309--4317},
  year={2020}
}
```

### Acknowledgements
Code is inspired by [MC-GAN](https://github.com/azadis/MC-GAN/blob/master/README.md).
Datasets are collected from [MC-GAN](https://github.com/azadis/MC-GAN/blob/master/README.md) and [Sandun.LK](https://sandunlk.home.blog/)