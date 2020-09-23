################################################################################
# MC-GAN
# Glyph Network Model
# By Samaneh Azadi
################################################################################

import torch
from collections import OrderedDict
from torch.autograd import Variable
import xifontgan.util.XIutil as util
from xifontgan.util.image_pool import ImagePool
from xifontgan.util.indexing import str2index
from .XIbase_model import BaseModel
from . import XInetworks
import random


class skGANModel(BaseModel):
    def name(self):
        return 'skGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.skmode = self.opt.skmode
        self.output_str_indices = str2index(self.opt.str_output,
                                            self.opt.charset)
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        if self.output_str_indices:
            self.str_image = self.Tensor(opt.batchSize, len(opt.str_output),
                                         opt.fineSize, opt.fineSize)
        if not self.skmode == 1:
            self.input_Ask0 = self.Tensor(opt.batchSize, opt.input_nc,
                                          opt.fineSize, opt.fineSize)
            self.input_Bsk0 = self.Tensor(opt.batchSize, opt.output_nc,
                                          opt.fineSize, opt.fineSize)
        if not self.skmode == 0:
            self.input_Ask1 = self.Tensor(opt.batchSize, opt.input_nc,
                                          opt.fineSize, opt.fineSize)
            self.input_Bsk1 = self.Tensor(opt.batchSize, opt.output_nc,
                                          opt.fineSize, opt.fineSize)

        # load/define networks
        if self.opt.conv3d:
            self.netG_3d = XInetworks.define_G_3d(
                opt.input_nc, opt.input_nc, norm=opt.norm,
                groups=opt.grps, gpu_ids=self.gpu_ids, ksize=7,
                    padding=3)

        self.netG = XInetworks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG,
                                        opt.norm, opt.use_dropout,
                                        gpu_ids=self.gpu_ids,
                                        ds_n=opt.downsampling_0_n,
                                        ds_mult=opt.downsampling_0_mult,
                                        ds_post=opt.dspost_G)
        if not self.skmode == 1:
            if self.opt.conv3d:
                self.netGsk0_3d = XInetworks.define_G_3d(
                    opt.input_nc, opt.input_nc, norm=opt.norm,
                    groups=opt.grps, gpu_ids=self.gpu_ids, ksize=9,
                    padding=4)
            self.netGsk0 = XInetworks.define_skG(
                opt.input_nc, opt.output_nc, opt.ngf,
                opt.which_model_netG, opt.norm, opt.use_dropout,
                gpu_ids=self.gpu_ids, ds_n=opt.downsampling_0_n,
                ds_mult=opt.downsampling_0_mult)
        if not self.skmode == 0:
            if self.opt.conv3d:
                self.netGsk1_3d = XInetworks.define_G_3d(
                    opt.input_nc, opt.input_nc, norm=opt.norm,
                    groups=opt.grps, gpu_ids=self.gpu_ids, ksize=9,
                    padding=4)
            self.netGsk1 = XInetworks.define_skG(
                opt.input_nc, opt.output_nc, opt.ngf,
                opt.which_model_netG, opt.norm, opt.use_dropout,
                gpu_ids=self.gpu_ids, ds_n=opt.downsampling_0_n,
                ds_mult=opt.downsampling_0_mult)
        
        disc_ch = opt.input_nc
        nif = disc_ch + disc_ch
        netD_norm = opt.norm
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.conditional:
                if opt.which_model_preNet != 'none':
                    self.preNet_A = XInetworks.define_preNet(
                        nif, nif,
                        which_model_preNet = opt.which_model_preNet,
                        norm=opt.norm, gpu_ids=self.gpu_ids)
                    if not self.skmode == 1:
                        self.preNet_Ask0 = XInetworks.define_preNet(
                            nif, nif,
                            which_model_preNet=opt.which_model_preNet,
                            norm=opt.norm, gpu_ids=self.gpu_ids)
                    if not self.skmode == 0:
                        self.preNet_Ask1 = XInetworks.define_preNet(
                            nif, nif,
                            which_model_preNet=opt.which_model_preNet,
                            norm=opt.norm, gpu_ids=self.gpu_ids)
                self.netD = XInetworks.define_D(nif, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, netD_norm, use_sigmoid, gpu_ids=self.gpu_ids)
                if not self.skmode == 1:
                    self.netDsk0 = XInetworks.define_D(
                        nif, opt.ndf, opt.which_model_netD,
                        opt.n_layers_D, netD_norm, use_sigmoid,
                        gpu_ids=self.gpu_ids)
                if not self.skmode == 0:
                    self.netDsk1 = XInetworks.define_D(
                        nif, opt.ndf, opt.which_model_netD,
                        opt.n_layers_D, netD_norm, use_sigmoid,
                        gpu_ids=self.gpu_ids)
            else:
                self.netD = XInetworks.define_D(
                    disc_ch, opt.ndf, opt.which_model_netD,
                    opt.n_layers_D, netD_norm, use_sigmoid,
                    gpu_ids=self.gpu_ids)
                if not self.skmode == 1:
                    self.netDsk0 = XInetworks.define_D(
                        disc_ch, opt.ndf, opt.which_model_netD,
                        opt.n_layers_D, netD_norm, use_sigmoid,
                        gpu_ids=self.gpu_ids)
                if not self.skmode == 0:
                    self.netDsk1 = XInetworks.define_D(
                        disc_ch, opt.ndf, opt.which_model_netD,
                        opt.n_layers_D, netD_norm, use_sigmoid,
                        gpu_ids=self.gpu_ids)
        latest = opt.continue_latest
        if not self.isTrain or (opt.which_epoch > 0):
            if self.opt.conv3d:
                self.load_network(self.netG_3d, 'G_3d', latest=latest)
                if not self.skmode == 1:
                    self.load_network(self.netGsk0_3d, 'Gsk0_3d',
                                      latest=latest)
                if not self.skmode == 0:
                    self.load_network(self.netGsk1_3d, 'Gsk1_3d',
                                      latest=latest)
            self.load_network(self.netG, 'G', latest=latest)
            if not self.skmode == 1:
                self.load_network(self.netGsk0, 'Gsk0', latest=latest)
            if not self.skmode == 0:
                self.load_network(self.netGsk1, 'Gsk1', latest=latest)
            if self.isTrain:
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', latest=latest)
                    if not self.skmode == 1:
                        self.load_network(self.preNet_Ask0,
                                          'PRE_Ask0', latest=latest)
                    if not self.skmode == 0:
                        self.load_network(self.preNet_Ask1,
                                          'PRE_Ask1', latest=latest)
                self.load_network(self.netD, 'D', latest=latest)
                if not self.skmode == 1:
                    self.load_network(
                        self.netDsk0, 'Dsk0', latest=latest)
                if not self.skmode == 0:
                    self.load_network(
                        self.netDsk1, 'Dsk1', latest=latest)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            if not self.skmode == 1:
                self.fake_sk0_pool = ImagePool(opt.pool_size)
            if not self.skmode == 0:
                self.fake_sk1_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = XInetworks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()

            # initialize optimizers
            if self.opt.conv3d:
                self.optimizer_G_3d = torch.optim.Adam(
                    self.netG_3d.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999))
                if not self.skmode == 1:
                    self.optimizer_Gsk0_3d = torch.optim.Adam(
                        self.netGsk0_3d.parameters(), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
                if not self.skmode == 0:
                    self.optimizer_Gsk1_3d = torch.optim.Adam(
                        self.netGsk1_3d.parameters(), lr=opt.lr,
                        betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr,
                betas=(opt.beta1, 0.999))
            if not self.skmode == 1:
                self.optimizer_Gsk0 = torch.optim.Adam(
                    self.netGsk0.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999))
            if not self.skmode == 0:
                self.optimizer_Gsk1 = torch.optim.Adam(
                    self.netGsk1.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999))

            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(
                    self.preNet_A.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999))
                if not self.skmode == 1:
                    self.optimizer_preAsk0 = torch.optim.Adam(
                        self.preNet_Ask0.parameters(), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
                if not self.skmode == 0:
                    self.optimizer_preAsk1 = torch.optim.Adam(
                        self.preNet_Ask1.parameters(), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr,
                betas=(opt.beta1, 0.999))
            if not self.skmode == 1:
                self.optimizer_Dsk0 = torch.optim.Adam(
                    self.netDsk0.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999))
            if not self.skmode == 0:
                self.optimizer_Dsk1 = torch.optim.Adam(
                    self.netDsk1.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            if self.opt.conv3d:
                XInetworks.print_network(self.netG_3d)
            XInetworks.print_network(self.netG)
            if opt.which_model_preNet != 'none':
                XInetworks.print_network(self.preNet_A)
            XInetworks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        if not self.skmode == 1:
            input_Ask0 = input['Ask0']
            input_Bsk0 = input['Bsk0']
            self.input_Ask0.resize_(input_Ask0.size()).copy_(input_Ask0)
            self.input_Bsk0.resize_(input_Bsk0.size()).copy_(input_Bsk0)
        if not self.skmode == 0:
            input_Ask1 = input['Ask1']
            input_Bsk1 = input['Bsk1']
            self.input_Ask1.resize_(input_Ask1.size()).copy_(input_Ask1)
            self.input_Bsk1.resize_(input_Bsk1.size()).copy_(input_Bsk1)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = self.input_A.detach()
        self.real_B = self.input_B.detach()
        if not self.skmode == 1:
            self.real_Ask0 = self.input_Ask0.detach()
            self.real_Bsk0 = self.input_Bsk0.detach()
        if not self.skmode == 0:
            self.real_Ask1 = self.input_Ask1.detach()
            self.real_Bsk1 = self.input_Bsk1.detach()
        # print('REAL_A size is')
        # print(self.real_A.size())
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
            if not self.skmode == 1:
                self.real_Ask0_indep = self.netGsk0_3d.forward(
                    self.real_Ask0.unsqueeze(2))
                self.fake_Bsk0 = self.netGsk0.forward(
                    self.real_Ask0_indep.squeeze(2))
            if not self.skmode == 0:
                self.real_Ask1_indep = self.netGsk1_3d.forward(
                    self.real_Ask1.unsqueeze(2))
                self.fake_Bsk1 = self.netGsk1.forward(
                    self.real_Ask1_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)
            if not self.skmode == 1:
                self.fake_Bsk0 = self.netGsk0.forward(self.real_Ask0)
            if not self.skmode == 1:
                self.fake_Bsk0 = self.netGsk0.forward(self.real_Ask0)
    
    def add_noise_disc(self, real):
        #add noise to the discriminator target labels
        #real: True/False?
        if self.opt.noisy_disc:
            rand_lbl = random.random()
            if rand_lbl<0.1:
                label = (not real)
            else:
                label = (real)
        else:  
            label = (real)
        return label
    
    # no backprop gradients
    def test(self):
        # for m in self.netG.modules():
        #     # if isinstance(m, torch.nn.BatchNorm2d) | isinstance(m, torch.nn.Dropout):
        #     m.eval()
        # for m in self.netG_3d.modules():
        #     # if isinstance(m, torch.nn.BatchNorm3d):
        #     m.eval()

        self.real_A = self.input_A.detach()
        self.real_B = self.input_B.detach()
        if not self.skmode == 1:
            self.real_Ask0 = self.input_Ask0.detach()
            self.real_Bsk0 = self.input_Bsk0.detach()
        if not self.skmode == 0:
            self.real_Ask1 = self.input_Ask1.detach()
            self.real_Bsk1 = self.input_Bsk1.detach()
        # print('REAL_A size is')
        # print(self.real_A.size())
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(
                self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(
                self.real_A_indep.squeeze(2))
            if not self.skmode == 1:
                self.real_Ask0_indep = self.netGsk0_3d.forward(
                    self.real_Ask0.unsqueeze(2))
                self.fake_Bsk0 = self.netGsk0.forward(
                    self.real_Ask0_indep.squeeze(2))
            if not self.skmode == 0:
                self.real_Ask1_indep = self.netGsk1_3d.forward(
                    self.real_Ask1.unsqueeze(2))
                self.fake_Bsk1 = self.netGsk1.forward(
                    self.real_Ask1_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)
            if not self.skmode == 1:
                self.fake_Bsk0 = self.netGsk0.forward(self.real_Ask0)
            if not self.skmode == 0:
                self.fake_Bsk1 = self.netGsk1.forward(self.real_Ask1)

        # self.criterionL1 = torch.nn.L1Loss()
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # print("L1 Loss is %f" % self.loss_G_L1)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)

        b,c,m,n = self.fake_B.size()
        rgb = 3 if self.opt.rgb else 1

        self.fake_B_reshaped = self.fake_B
        self.real_A_reshaped = self.real_A
        self.real_B_reshaped = self.real_B

        if self.opt.conditional:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
            self.pred_fake_patch = self.netD.forward(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake_patch, label_fake)
            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_AB = self.preNet_A.forward(fake_AB.detach())
                self.pred_fake = self.netD.forward(transformed_AB)
                self.loss_D_fake += self.criterionGAN(self.pred_fake, label_fake)
            if not self.skmode == 1:
                fake_sk0 = self.fake_sk0_pool.query(torch.cat(
                    (self.fake_Bsk0, self.fake_B), 1))
                self.pred_fake_sk0_patch = self.netDsk0.forward(
                    fake_sk0.detach())
                self.loss_Dsk0_fake = self.criterionGAN(
                    self.pred_fake_sk0_patch, label_fake)
                if self.opt.which_model_preNet != 'none':
                    # transform the input
                    transformed_sk0_fake = self.preNet_Ask0.forward(
                        fake_sk0.detach())
                    self.pred_fake_sk0 = self.netDsk0.forward(
                        transformed_sk0_fake)
                    self.loss_Dsk0_fake += self.criterionGAN(
                        self.pred_fake_sk0, label_fake)
            if not self.skmode == 0:
                fake_sk1 = self.fake_sk1_pool.query(torch.cat(
                    (self.fake_Bsk1, self.fake_B), 1))
                self.pred_fake_sk1_patch = self.netDsk1.forward(
                    fake_sk1.detach())
                self.loss_Dsk1_fake = self.criterionGAN(
                    self.pred_fake_sk1_patch, label_fake)
                if self.opt.which_model_preNet != 'none':
                    # transform the input
                    transformed_sk1_fake = self.preNet_Ask1.forward(
                        fake_sk1.detach())
                    self.pred_fake_sk1 = self.netDsk1.forward(
                        transformed_sk1_fake)
                    self.loss_Dsk1_fake += self.criterionGAN(
                        self.pred_fake_sk1, label_fake)
        else:
            self.pred_fake = self.netD.forward(self.fake_B.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, label_fake)
            if not self.skmode == 1:
                self.loss_Dsk0_fake = 0
            if not self.skmode == 0:
                self.loss_Dsk1_fake = 0

        # Real
        label_real = self.add_noise_disc(True)
        if self.opt.conditional:
            real_AB = torch.cat((self.real_A_reshaped, self.real_B_reshaped), 1).detach()
            self.pred_real_patch = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real_patch, label_real)
            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_A_real = self.preNet_A.forward(real_AB)
                self.pred_real = self.netD.forward(transformed_A_real)
                self.loss_D_real += self.criterionGAN(self.pred_real, label_real)
            if not self.skmode == 1:
                real_sk0 = torch.cat((self.real_Bsk0,
                                      self.real_B), 1).detach()
                self.pred_real_sk0_patch = self.netDsk0.forward(
                    real_sk0.detach())
                self.loss_Dsk0_real = self.criterionGAN(
                    self.pred_real_sk0_patch, label_real)
                if self.opt.which_model_preNet != 'none':
                    # transform the input
                    transformed_sk0_real = self.preNet_Ask0.forward(
                        real_sk0.detach())
                    self.pred_real_sk0 = self.netDsk0.forward(
                        transformed_sk0_real)
                    self.loss_Dsk0_real += self.criterionGAN(
                        self.pred_real_sk0, label_real)
            if not self.skmode == 0:
                real_sk1 = torch.cat((self.real_Bsk1,
                                      self.real_B), 1).detach()
                self.pred_real_sk1_patch = self.netDsk1.forward(
                    real_sk1.detach())
                self.loss_Dsk1_real = self.criterionGAN(
                    self.pred_real_sk1_patch, label_real)
                if self.opt.which_model_preNet != 'none':
                    # transform the input
                    transformed_sk1_real = self.preNet_Ask1.forward(
                        real_sk1.detach())
                    self.pred_real_sk1 = self.netDsk1.forward(
                        transformed_sk1_real)
                    self.loss_Dsk1_real += self.criterionGAN(
                        self.pred_real_sk1, label_real)
        else:
            self.pred_real = self.netD.forward(self.real_B)            
            self.loss_D_real = self.criterionGAN(self.pred_real, label_real)
            if not self.skmode == 1:
                self.loss_Dsk0_real = 0
            if not self.skmode == 0:
                self.loss_Dsk1_real = 0
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if not self.skmode == 1:
            self.loss_D += (self.loss_Dsk0_fake +
                            self.loss_Dsk0_real) * 0.5
        if not self.skmode == 0:
            self.loss_D += (self.loss_Dsk1_fake +
                            self.loss_Dsk1_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.conditional:
            #PATCH GAN
            fake_AB = torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1)
            pred_fake_patch = self.netD.forward(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
            if self.opt.which_model_preNet != 'none':
                #global disc
                transformed_A = self.preNet_A.forward(fake_AB)
                pred_fake = self.netD.forward(transformed_A)
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
            if not self.skmode == 1:
                fake_sk0 = torch.cat((self.fake_Bsk0, self.fake_B), 1)
                pred_fake_sk0_patch = self.netDsk0.forward(fake_sk0)
                self.loss_Gsk0_GAN = self.criterionGAN(
                    pred_fake_sk0_patch, True)
                if self.opt.which_model_preNet != 'none':
                    # global disc
                    transformed_Ask0 = self.preNet_Ask0.forward(
                        fake_sk0)
                    pred_fake_sk0 = self.netDsk0.forward(
                        transformed_Ask0)
                    self.loss_Gsk0_GAN += self.criterionGAN(
                        pred_fake_sk0, True)
            if not self.skmode == 0:
                fake_sk1 = torch.cat((self.fake_Bsk1, self.fake_B), 1)
                pred_fake_sk1_patch = self.netDsk1.forward(fake_sk1)
                self.loss_Gsk1_GAN = self.criterionGAN(
                    pred_fake_sk1_patch, True)
                if self.opt.which_model_preNet != 'none':
                    # global disc
                    transformed_Ask1 = self.preNet_Ask1.forward(
                        fake_sk1)
                    pred_fake_sk1 = self.netDsk1.forward(
                        transformed_Ask1)
                    self.loss_Gsk1_GAN += self.criterionGAN(
                        pred_fake_sk1, True)
        else:
            pred_fake = self.netD.forward(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            if not self.skmode == 1:
                self.loss_Gsk0_GAN = 0
            if not self.skmode == 0:
                self.loss_Gsk1_GAN = 0

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if not self.skmode == 1:
            self.loss_Gsk0_L2 = self.criterionMSE(
                self.fake_Bsk0, self.real_Bsk0) * self.opt.lambda_A\
                                * 10
            self.loss_G += self.loss_Gsk0_L2 + self.loss_Gsk0_GAN
        if not self.skmode == 0:
            self.loss_Gsk1_L2 = self.criterionMSE(
                self.fake_Bsk1, self.real_Bsk1) * self.opt.lambda_A\
                                * 10
            self.loss_G += self.loss_Gsk1_L2 + self.loss_Gsk1_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        if not self.skmode == 1:
            self.optimizer_Dsk0.zero_grad()
        if not self.skmode == 0:
            self.optimizer_Dsk1.zero_grad()

        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.zero_grad()
            if not self.skmode == 1:
                self.optimizer_preAsk0.zero_grad()
            if not self.skmode == 0:
                self.optimizer_preAsk1.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        if not self.skmode == 1:
            self.optimizer_Dsk0.step()
        if not self.skmode == 0:
            self.optimizer_Dsk1.step()

        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.step()
            if not self.skmode == 1:
                self.optimizer_preAsk0.step()
            if not self.skmode == 0:
                self.optimizer_preAsk1.step()

        self.optimizer_G.zero_grad()
        if not self.skmode == 1:
            self.optimizer_Gsk0.zero_grad()
        if not self.skmode == 0:
            self.optimizer_Gsk1.zero_grad()

        if self.opt.conv3d:
            self.optimizer_G_3d.zero_grad()
            if not self.skmode == 1:
                self.optimizer_Gsk0_3d.zero_grad()
            if not self.skmode == 0:
                self.optimizer_Gsk1_3d.zero_grad()

        self.backward_G()

        self.optimizer_G.step()
        if not self.skmode == 1:
            self.optimizer_Gsk0.step()
        if not self.skmode == 0:
            self.optimizer_Gsk1.step()

        if self.opt.conv3d:
            self.optimizer_G_3d.step()
            if not self.skmode == 1:
                self.optimizer_Gsk0_3d.step()
            if not self.skmode == 0:
                self.optimizer_Gsk1_3d.step()

    def get_current_errors(self):
        errors = [('G_GAN', self.loss_G_GAN.data),
                  ('G_L1', self.loss_G_L1.data),
                  ('D_real', self.loss_D_real.data),
                  ('D_fake', self.loss_D_fake.data)]
        if not self.skmode == 1:
            errors += [('Gsk0_L2', self.loss_Gsk0_L2),
                       ('Gsk0_GAN', self.loss_Gsk0_GAN),
                       ('Dsk0_real', self.loss_Dsk0_real),
                       ('Dsk0_fake', self.loss_Dsk0_fake)]
        if not self.skmode == 0:
            errors += [('Gsk1_L2', self.loss_Gsk1_L2),
                       ('Gsk1_GAN', self.loss_Gsk1_GAN),
                       ('Dsk1_real', self.loss_Dsk1_real),
                       ('Dsk1_fake', self.loss_Dsk1_fake)]
        return OrderedDict(errors)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.output_str_indices:
            output_str_indices = torch.LongTensor(self.output_str_indices)
            output_str_data = self.fake_B.data[:, output_str_indices, :, :]
            real_str_data = self.real_B.data[:, output_str_indices, :, :]
            img_output_str = util.tensor2im(output_str_data)
            img_real_str = util.tensor2im(real_str_data)
            list = [('real_A', real_A), ('fake_B', fake_B),
                    ('real_B', real_B), ('fake_STR', img_output_str),
                    ('real_STR', img_real_str)]
        else:
            list = [('real_A', real_A), ('fake_B', fake_B),
                    ('real_B', real_B)]
        if not self.skmode == 1:
            real_Ask0 = util.tensor2im(self.real_Ask0.data)
            fake_Bsk0 = util.tensor2im(self.fake_Bsk0.data)
            real_Bsk0 = util.tensor2im(self.real_Bsk0.data)
            list += [('real_Ask0', real_Ask0), ('real_Bsk0', real_Bsk0),
                     ('fake_Bsk0', fake_Bsk0)]
        if not self.skmode == 0:
            real_Ask1 = util.tensor2im(self.real_Ask1.data)
            fake_Bsk1 = util.tensor2im(self.fake_Bsk1.data)
            real_Bsk1 = util.tensor2im(self.real_Bsk1.data)
            list += [('real_Ask1', real_Ask1), ('real_Bsk1', real_Bsk1),
                     ('fake_Bsk1', fake_Bsk1)]
        return OrderedDict(list)

    def save(self, latest=False):
        if self.opt.conv3d:
            self.save_network(self.netG_3d, 'G_3d',
                               gpu_ids=self.gpu_ids, latest=latest)
            if not self.skmode == 1:
                self.save_network(self.netGsk0_3d, 'Gsk0_3d',
                                  gpu_ids=self.gpu_ids, latest=latest)
            if not self.skmode == 0:
                self.save_network(self.netGsk1_3d, 'Gsk1_3d',
                                  gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netG, 'G', gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netD, 'D', gpu_ids=self.gpu_ids, latest=latest)
        if not self.skmode == 1:
            self.save_network(self.netGsk0, 'Gsk0',
                              gpu_ids=self.gpu_ids,
                              latest=latest)
            self.save_network(self.netDsk0, 'Dsk0',
                              gpu_ids=self.gpu_ids,
                              latest=latest)
        if not self.skmode == 0:
            self.save_network(self.netGsk1, 'Gsk1',
                              gpu_ids=self.gpu_ids,
                              latest=latest)
            self.save_network(self.netDsk1, 'Dsk1',
                              gpu_ids=self.gpu_ids,
                              latest=latest)

        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A', gpu_ids=self.gpu_ids, latest=latest)
            if not self.skmode == 1:
                self.save_network(self.preNet_Ask0, 'PRE_Ask0',
                                  gpu_ids=self.gpu_ids, latest=latest)
            if not self.skmode == 0:
                self.save_network(self.preNet_Ask1, 'PRE_Ask1',
                                  gpu_ids=self.gpu_ids, latest=latest)

    def set_learning_rate(self, epoch):
        lr = self.opt.lr / self.batch_skip
        if epoch > self.opt.niter:
            lr = lr * (self.opt.niter + self.opt.niter_decay -
                       epoch) / self.opt.niter_decay
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.which_model_preNet != 'none':
            for param_group in self.optimizer_preA.param_groups:
                param_group['lr'] = lr
        if self.opt.conv3d:
            for param_group in self.optimizer_G_3d.param_groups:
                param_group['lr'] = lr

        if not self.skmode == 1:
            for param_group in self.optimizer_Dsk0.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_Gsk0.param_groups:
                param_group['lr'] = lr
            if self.opt.which_model_preNet != 'none':
                for param_group in self.optimizer_preAsk0.param_groups:
                    param_group['lr'] = lr
            if self.opt.conv3d:
                for param_group in self.optimizer_Gsk0_3d.param_groups:
                    param_group['lr'] = lr
        if not self.skmode == 0:
            for param_group in self.optimizer_Dsk1.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_Gsk1.param_groups:
                param_group['lr'] = lr
            if self.opt.which_model_preNet != 'none':
                for param_group in \
                        self.optimizer_preAsk1.param_groups:
                    param_group['lr'] = lr
            if self.opt.conv3d:
                for param_group in \
                        self.optimizer_Gsk1_3d.param_groups:
                    param_group['lr'] = lr


        print('setting learning rate: %f' % lr)

    # def update_learning_rate(self):
    #     lrd = self.opt.lr / self.opt.niter_decay
    #     lr = self.old_lr - lrd
    #     for param_group in self.optimizer_D.param_groups:
    #         param_group['lr'] = lr
    #     if self.opt.which_model_preNet != 'none':
    #         for param_group in self.optimizer_preA.param_groups:
    #             param_group['lr'] = lr
    #     for param_group in self.optimizer_G.param_groups:
    #         param_group['lr'] = lr
    #     if self.opt.conv3d:
    #         for param_group in self.optimizer_G_3d.param_groups:
    #             param_group['lr'] = lr
    #     print('update learning rate: %f -> %f' % (self.old_lr, lr))
    #     self.old_lr = lr
