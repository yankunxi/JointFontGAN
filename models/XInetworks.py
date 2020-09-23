################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import functools
# from XIResnet import *
from torch import index_select


# , LongTensor


def weights_init(m):
    classname = m.__class__.__name__
    print("classname", classname)
    if classname.find('Conv') != -1:
        print("in random conv")
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        print("in random batchnorm")
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=False)
    else:
        norm_layer = None
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def conv_norm_relu_module(norm_layer, input_nc, ngf, kernel_size,
                          padding, stride=1, relu='relu'):
    model = [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size,
                       padding=padding, stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]

    if relu == 'relu':
        model += [nn.ReLU(True)]
    elif relu == 'Lrelu':
        model += [nn.LeakyReLU(0.2, True)]

    return model


def relu_conv_norm_module(norm_layer, input_nc, ngf, kernel_size,
                          padding, stride=1, relu='relu'):
    if relu == 'relu':
        model = [nn.ReLU(True)]
    elif relu == 'Lrelu':
        model = [nn.LeakyReLU(0.2, True)]

    model += [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size,
                        padding=padding, stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]

    return model


def convTranspose_norm_relu_module(norm_layer, input_nc, ngf,
                                   kernel_size, padding, stride=1,
                                   output_padding=0):
    if norm_layer:
        model = [nn.ConvTranspose2d(input_nc, ngf,
                                    kernel_size=kernel_size,
                                    stride=stride, padding=padding,
                                    output_padding=output_padding),
                 norm_layer(int(ngf)),
                 nn.ReLU(True)]

    return model


def define_G_3d(input_nc, output_nc, norm='batch', groups=26, ksize=3,
                padding=1, gpu_ids=[]):
    netG_3d = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    netG_3d = ResnetGenerator_3d_conv(input_nc, output_nc,
                                      norm_type=norm,
                                      groups=groups, ksize=ksize,
                                      padding=padding,
                                      gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        netG_3d.cuda(device=gpu_ids[0])
    netG_3d.apply(weights_init)
    return netG_3d


def define_G(input_nc, output_nc, ngf, which_model_netG,
             norm='batch', use_dropout=False, gpu_ids=[],
             ds_n=2, ds_mult=3, ds_post=0):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf,
                               norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, ds_post=ds_post)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf,
                               norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, ds_n=ds_n,
                               ds_mult=ds_mult, ds_post=ds_post)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'uresnet_64':
        netG = UResnetGenerator(input_nc, output_nc, 6, ngf,
                                norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                gpu_ids=gpu_ids)
    else:
        print(
            'Generator model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_skG(input_nc, output_nc, ngf, which_model_netG,
               norm='batch', use_dropout=False, gpu_ids=[],
               ds_n=2, ds_mult=3, ds_post=0, skmode=0):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    if which_model_netG == 'resnet_9blocks':
        netG = skResnetGenerator(input_nc, output_nc, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=9,
                                 gpu_ids=gpu_ids, ds_post=ds_post,
                                 skmode=skmode)
    elif which_model_netG == 'resnet_6blocks':
        netG = skResnetGenerator(input_nc, output_nc, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=6,
                                 gpu_ids=gpu_ids, ds_n=ds_n,
                                 ds_mult=ds_mult, ds_post=ds_post,
                                 skmode=skmode)
    # elif which_model_netG == 'unet_128':
    #     netG = skUnetGenerator(input_nc, output_nc, 7, ngf,
    #                          norm_layer=norm_layer,
    #                          use_dropout=use_dropout, gpu_ids=gpu_ids, skmode=skmode)
    # elif which_model_netG == 'unet_64':
    #     netG = skUnetGenerator(input_nc, output_nc, 6, ngf,
    #                          norm_layer=norm_layer,
    #                          use_dropout=use_dropout, gpu_ids=gpu_ids, skmode=skmode)
    # elif which_model_netG == 'uresnet_64':
    #     netG = skUResnetGenerator(input_nc, output_nc, 6, ngf,
    #                             norm_layer=norm_layer,
    #                             use_dropout=use_dropout,
    #                             gpu_ids=gpu_ids, skmode=skmode)
    else:
        print(
            'Generator model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_LSTMG(input_nc, output_nc, batch_size, fine_size,
                 remainnum, ngf, hidden_size, which_model_netG,
                 norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)
    LSTM_core = LSTMcore(batch_size, hidden_size)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    # if which_model_netG == 'resnet_9blocks':
    #     netG = ResnetLSTMGenerator(input_nc, output_nc, batch_size,
    #                                fine_size, remainnum, ngf,
    #                                norm_layer=norm_layer,
    #                                use_dropout=use_dropout, n_blocks=9,
    #                                gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_6blocks':
    #     netG = ResnetLSTMGenerator(input_nc, output_nc, batch_size,
    #                                fine_size, remainnum, ngf,
    #                                norm_layer=norm_layer,
    #                                use_dropout=use_dropout, n_blocks=6,
    #                                gpu_ids=gpu_ids)
    if which_model_netG == 'unet_128':
        netG = UnetLSTMGenerator(input_nc, output_nc, 7, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout,
                                 gpu_ids=gpu_ids, LSTM_core=LSTM_core)
    elif which_model_netG == 'unet_64':
        netG = UnetLSTMGenerator(input_nc, output_nc, 6, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout,
                                 gpu_ids=gpu_ids, LSTM_core=LSTM_core)
    else:
        print('LSTM Generator model name [%s] is not recognized' %
              which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_Enc(input_nc, output_nc, ngf, which_model_netG,
               norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetEncoder(input_nc, output_nc, ngf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout, n_blocks=9,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetEncoder(input_nc, output_nc, ngf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout, n_blocks=6,
                             gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetEncoder(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_64':
    #     netG = UnetEncoder(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print(
            'encoder model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_Dec(input_nc, output_nc, ngf, which_model_netG,
               norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetDecoder(input_nc, output_nc, ngf,
                             norm_type='batch',
                             use_dropout=use_dropout, n_blocks=9,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetDecoder(input_nc, output_nc, ngf,
                             norm_type='batch',
                             use_dropout=use_dropout, n_blocks=6,
                             gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetDecoder(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_64':
    #     netG = UnetDecoder(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print(
            'Decoder model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_LSTMEnc(input_nc, output_nc, batch_size, fine_size,
                   remainnum,
                   ngf, which_model_netG, norm='batch',
                   use_dropout=False,
                   gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetLSTMEncoder(input_nc, output_nc, batch_size,
                                 fine_size,
                                 remainnum, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=9,
                                 gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetLSTMEncoder(input_nc, output_nc, batch_size,
                                 fine_size,
                                 remainnum, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=6,
                                 gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetEncoder(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_64':
    #     netG = UnetEncoder(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print(
            'encoder model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_LSTMDec(input_nc, output_nc, batch_size, fine_size,
                   remainnum,
                   ngf, which_model_netG, norm='batch',
                   use_dropout=False,
                   gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    print(which_model_netG)
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetLSTMDecoder(input_nc, output_nc, batch_size,
                                 fine_size,
                                 remainnum, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=9,
                                 norm_type=norm, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetLSTMDecoder(input_nc, output_nc, batch_size,
                                 fine_size,
                                 remainnum, ngf,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=6,
                                 norm_type=norm, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetDecoder(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_64':
    #     netG = UnetDecoder(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print(
            'Decoder model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False,
             postConv=True, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3,
                                   use_sigmoid=use_sigmoid,
                                   norm_layer=norm_layer,
                                   postConv=postConv, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D,
                                   use_sigmoid=use_sigmoid,
                                   norm_layer=norm_layer,
                                   postConv=postConv, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_preNet(input_nc, nif=32, which_model_preNet='none',
                  norm='batch', gpu_ids=[]):
    preNet = None
    norm_layer = get_norm_layer(norm_type=norm)
    use_gpu = len(gpu_ids) > 0
    if which_model_preNet == '2_layers':
        print(
            "2 layers convolution applied before being fed into the discriminator")
        preNet = InputTransformation(input_nc, nif, norm_layer,
                                     gpu_ids)
        if use_gpu:
            assert (torch.cuda.is_available())
            preNet.cuda(device=gpu_ids[0])
        preNet.apply(weights_init)
    return preNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class View4LSTM(nn.Module):
    def __init__(self, submodule=None):
        super(View4LSTM, self).__init__()
        model = submodule

    def forward(self, x):
        shape = list([x.size()[0], -1])
        return x.view(shape)


class ImageLSTM(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        shape = list(x.size())[0:2] + [-1]
        return x.view(shape)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (
                                        self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(
                    self.real_label)
                self.real_label_var = Variable(real_tensor,
                                               requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (
                                        self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(
                    self.fake_label)
                self.fake_label_var = Variable(fake_tensor,
                                               requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the first layer of the generator to make different glyphs independent
# conv3D is used instead of conv2D.
class ResnetGenerator_3d_conv(nn.Module):
    def __init__(self, input_nc, output_nc, norm_type='batch',
                 groups=26, ksize=3, padding=1, gpu_ids=[]):
        super(ResnetGenerator_3d_conv, self).__init__()
        self.input_nc = input_nc
        self.gpu_ids = gpu_ids
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm3d,
                                           affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm3d,
                                           affine=False)
        else:
            norm_layer = functools.partial(nn.BatchNorm3d,
                                           affine=True)
        # norm_layer = get_norm_layer(norm_type=norm_type)

        model = [nn.Conv3d(input_nc, output_nc, kernel_size=ksize,
                           padding=padding, groups=groups),
                 norm_layer(output_nc), nn.ReLU(True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


# Defines the decoder that consists of Resnet blocks between a few
# upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False,
                 n_blocks=6, norm_type='batch', gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        n_downsampling = 2
        factor_ch = 3
        mult = factor_ch ** n_downsampling
        model = []

        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d,
                                           affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d,
                                           affine=False)
        else:
            norm_layer = functools.partial(nn.BatchNorm2d,
                                           affine=True)

        for i in range(n_downsampling):
            mult = factor_ch ** (n_downsampling - i)

            model += convTranspose_norm_relu_module(norm_layer,
                                                    ngf * mult, int(
                    ngf * mult / factor_ch), 3, 1, stride=2,
                                                    output_padding=1)

        if norm_type == 'batch' or norm_type == 'instance':
            model += [
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert ('norm not defined')

        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


# Defines the encoder that consists of Resnet blocks between a few
# downsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = conv_norm_relu_module(norm_layer, input_nc, ngf, 7, 3)

        n_downsampling = 2
        for i in range(n_downsampling):
            factor_ch = 3  # 2**i : 3**i is a more complicated filter
            mult = factor_ch ** i
            model += conv_norm_relu_module(norm_layer, ngf * mult,
                                           ngf * mult * factor_ch, 3,
                                           1, stride=2)

        mult = factor_ch ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer,
                            use_dropout=use_dropout)]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)

        # Defines the generator that consists of Resnet blocks between a few


# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/


class ResnetLSTMEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, batch_size, fine_size,
                 remainnum,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetLSTMEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = []
        # model += View([-1, input_nc, fine_size, fine_size * 3])
        model += conv_norm_relu_module(norm_layer, input_nc, ngf,
                                       7, 3)

        n_downsampling = 2
        for i in range(n_downsampling):
            factor_ch = 3  # 2**i : 3**i is a more complicated filter
            mult = factor_ch ** i
            model += conv_norm_relu_module(norm_layer, ngf * mult,
                                           ngf * mult * factor_ch, 3,
                                           1, stride=2)

        mult = factor_ch ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer,
                            use_dropout=use_dropout)]

        # model += View([remainnum, batch_size, -1])
        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)

        # Defines the generator that consists of Resnet blocks between a few


class ResnetLSTMDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, batch_size, fine_size,
                 remainnum,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, norm_type='batch', gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetLSTMDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        n_downsampling = 2
        factor_ch = 3
        mult = factor_ch ** n_downsampling

        model = []
        # model += View([-1, input_nc, fine_size, fine_size * 3])
        for i in range(n_downsampling):
            mult = factor_ch ** (n_downsampling - i)

            model += convTranspose_norm_relu_module(norm_layer,
                                                    ngf * mult,
                                                    int(
                                                        ngf * mult / factor_ch),
                                                    3, 1, stride=2,
                                                    output_padding=1)

        if norm_layer:
            model += [
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert ('norm not defined')

        model += [nn.Tanh()]
        # model += View(remainnum, batch_size, input_nc, fine_size, fine_size * 3)
        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, gpu_ids=[], ds_n=2, ds_mult=3,
                 ds_post=0):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = conv_norm_relu_module(norm_layer, input_nc, ngf, 7, 3)

        # ds_n: number of downsampling layers
        # ds_mult: 2**i : 3**i is a more complicated filter
        for i in range(ds_post):
            conv_norm_relu_module(norm_layer, ngf, ngf, 7, 3)

        for i in range(ds_n):
            mult = ds_mult ** i
            model += conv_norm_relu_module(norm_layer, ngf * mult,
                                           ngf * mult * ds_mult, 3, 1,
                                           stride=2)

        mult = ds_mult ** ds_n
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer,
                            use_dropout=use_dropout)]

        for i in range(ds_n):
            mult = ds_mult ** (ds_n - i)
            model += convTranspose_norm_relu_module(norm_layer,
                                                    ngf * mult, int(
                    ngf * mult / ds_mult), 3, 1, stride=2,
                                                    output_padding=1)

        for i in range(ds_post):
            model += conv_norm_relu_module(norm_layer, ngf, ngf, 7, 3)

        if norm_layer:
            model += [
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert ('norm not defined')

        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, encoder=False):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


class skResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, gpu_ids=[], ds_n=2, ds_mult=3,
                 ds_post=0, skmode=1):
        assert (n_blocks >= 0)
        super(skResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = conv_norm_relu_module(norm_layer, input_nc, ngf, 7, 3)

        for i in range(ds_post):
            model += conv_norm_relu_module(norm_layer, ngf, ngf, 7, 3)

        # ds_n: number of downsampling layers
        # ds_mult: 2**i : 3**i is a more complicated filter
        for i in range(ds_n):
            mult = ds_mult ** i
            model += conv_norm_relu_module(norm_layer, ngf * mult,
                                           ngf * mult * ds_mult, 7,
                                           3, stride=2)

        mult = ds_mult ** ds_n
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer,
                            use_dropout=use_dropout)]

        for i in range(ds_n):
            mult = ds_mult ** (ds_n - i)
            model += convTranspose_norm_relu_module(
                norm_layer, ngf * mult, int(ngf * mult / ds_mult),
                7, 3, stride=2, output_padding=1)

        for i in range(ds_post):
            model += conv_norm_relu_module(norm_layer, ngf, ngf, 7, 3)

        if norm_layer:
            model += [
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert ('norm not defined')

        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, encoder=False):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


class ResnetLSTMGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, batch_size, fine_size,
                 remainnum,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetLSTMGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = View(
            [batch_size * remainnum, input_nc, fine_size, fine_size *
             3])
        model += conv_norm_relu_module(norm_layer, input_nc, ngf,
                                       7, 3)
        n_downsampling = 2
        for i in range(n_downsampling):
            factor_ch = 3  # 2**i : 3**i is a more complicated filter
            mult = factor_ch ** i
            model += conv_norm_relu_module(norm_layer, ngf * mult,
                                           ngf * mult * factor_ch, 3,
                                           1,
                                           stride=2)

        mult = factor_ch ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer,
                            use_dropout=use_dropout)]

        model += View([remainnum, batch_size, -1])
        model += nn.LSTM()

        for i in range(n_downsampling):
            mult = factor_ch ** (n_downsampling - i)

            model += convTranspose_norm_relu_module(norm_layer,
                                                    ngf * mult,
                                                    int(
                                                        ngf * mult / factor_ch),
                                                    3, 1,
                                                    stride=2,
                                                    output_padding=1)

        if norm_layer:
            model += [
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert ('norm not defined')

        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, encoder=False):

        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
                                                norm_layer,
                                                use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer,
                         use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm

        conv_block += conv_norm_relu_module(norm_layer, dim, dim,
                                            3, p)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.0)]

        # if norm_type=='batch' or norm_type=='instance':
        if norm_layer:
            conv_block += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                norm_layer(dim)]
        else:
            assert ("norm not defined")

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            # print(x.size())
            y = self.model(x)
            # print(y.size())
            return torch.cat([x, y], 1)


class UResnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UResnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        downres = ResnetBlock(inner_nc, 'zero',
                              norm_layer=norm_layer,
                              use_dropout=use_dropout)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        upres = ResnetBlock(outer_nc, 'zero',
                            norm_layer=norm_layer,
                            use_dropout=use_dropout)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downres]
            up = [uprelu, upconv, nn.Tanh(), upres]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm, upres]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, downres]
            up = [uprelu, upconv, upnorm, upres]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            # print(x.size())
            y = self.model(x)
            # print(y.size())
            return torch.cat([x, y], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 gpu_ids=[]):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.gpu_ids = gpu_ids
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                             input_nc=None,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(
                num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                                 input_nc=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf,
                                             input_nc=input_nc,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


class UResnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 gpu_ids=[]):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UResnetGenerator, self).__init__()
        # construct unet structure
        self.gpu_ids = gpu_ids
        unet_block = UResnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                                input_nc=None,
                                                submodule=None,
                                                norm_layer=norm_layer,
                                                innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers
            # with ngf * 8 filters
            unet_block = UResnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                                    input_nc=None,
                                                    submodule=unet_block,
                                                    norm_layer=norm_layer,
                                                    use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UResnetSkipConnectionBlock(ngf * 4, ngf * 8,
                                                input_nc=None,
                                                submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = UResnetSkipConnectionBlock(ngf * 2, ngf * 4,
                                                input_nc=None,
                                                submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = UResnetSkipConnectionBlock(ngf, ngf * 2,
                                                input_nc=None,
                                                submodule=unet_block,
                                                norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf,
                                             input_nc=input_nc,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


class LSTMcore(nn.Module):
    def __init__(self, batchSize, hidden_size):
        super(LSTMcore, self).__init__()
        self.batchSize = batchSize
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.init_h = Parameter(torch.randn(self.hidden_size).cuda(),
                                requires_grad=True)
        self.init_c = Parameter(torch.randn(self.hidden_size).cuda(),
                                requires_grad=True)
        self.h = self.init_h.repeat(self.batchSize, 1)
        self.c = self.init_c.repeat(self.batchSize, 1)
        # print("core init is " + str(sum(self.h)))

    def start(self):
        self.h = self.init_h.repeat(self.batchSize, 1)
        self.c = self.init_c.repeat(self.batchSize, 1)
        # print("core starting is " + str(sum(self.h)))

    def forward(self, input):
        print(self.lstm.h)
        shape = list([input.size()[0], -1])
        x = input.view(shape)
        self.h, self.c = self.lstm.forward(x, (self.h, self.c))
        y = self.h.view(input.size())
        print("core forward is " + str(sum(self.h)))
        return y


class UnetLSTMGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 gpu_ids=[], LSTM_core=None):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetLSTMGenerator, self).__init__()
        # construct unet structure
        self.gpu_ids = gpu_ids
        self.LSTM_core = LSTM_core
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                             input_nc=None,
                                             submodule=self.LSTM_core,
                                             norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                                 input_nc=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf,
                                             input_nc=input_nc,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        # print("core forward is " + str(sum(self.LSTM_core.h)))
        print("core forward is " + str(self.LSTM_core.lstm.h))
        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)


# Apply a transformation on the input and prediction before feeding into the discriminator
# in the conditional case
class InputTransformation(nn.Module):
    def __init__(self, input_nc, nif=32, norm_layer=nn.BatchNorm2d,
                 gpu_ids=[]):
        super(InputTransformation, self).__init__()
        self.gpu_ids = gpu_ids
        use_gpu = len(gpu_ids) > 0
        if use_gpu:
            assert (torch.cuda.is_available())

        sequence = [nn.Conv2d(input_nc, nif, kernel_size=3, stride=2,
                              padding=1),
                    norm_layer(nif),
                    nn.ReLU(True)]
        sequence += [
            nn.Conv2d(nif, nif, kernel_size=3, stride=2, padding=1),
            norm_layer(nif),
            nn.ReLU(True)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data,
                                            torch.cuda.FloatTensor):
            netTrans = nn.parallel.data_parallel(self.model, input,
                                                 self.gpu_ids)
        else:
            netTrans = self.model(input)
        return netTrans


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 postConv=True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 5
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
                      padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += conv_norm_relu_module(norm_layer,
                                              ndf * nf_mult_prev,
                                              ndf * nf_mult, kw, padw,
                                              stride=2, relu='Lrelu')

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += conv_norm_relu_module(norm_layer,
                                          ndf * nf_mult_prev,
                                          ndf * nf_mult, kw, padw,
                                          stride=1, relu='Lrelu')

        if postConv:
            sequence += [
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
                          padding=padw)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data,
                                            torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input,
                                             self.gpu_ids)
        else:
            return self.model(input)
