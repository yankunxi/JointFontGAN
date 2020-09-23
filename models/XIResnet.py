################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Yankun Xi
################################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import functools

