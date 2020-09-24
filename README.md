# JointFontGAN in PyTorch

This is the implementation of the [](https://arxiv.org/abs/). The code was written by [](https://github.com/).
If you use this code or our [](https://github.com/) for your research, please cite:


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
cd xifontgan
```

### JointFontGAN train/test


edit ./configure/device


```bash
. ./scripts/EskGAN/XItrain_EskGAN.sh Capitals64
```

```bash
. ./scripts/EskGAN/XItest_EskGAN.sh Capitals64 test
```
