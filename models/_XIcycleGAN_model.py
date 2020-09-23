################################################################################
# MC-GAN
# Glyph Network Model
# By Samaneh Azadi
# Modified by Yankun Xi
################################################################################

import torch
from collections import OrderedDict
from torch.autograd import Variable
import xifontgan.util.XIutil as util
from xifontgan.util.image_pool import ImagePool
from .XIbase_model import BaseModel
from . import XInetworks
import random


class cycleGANModel(BaseModel):
    def name(self):
        return 'cycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        # backward network has postfix g
        if self.opt.conv3d:
            self.netG_3d = XInetworks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)
            self.netGg_3d = XInetworks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)

        self.netG = XInetworks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG,
                                        opt.norm, opt.use_dropout,
                                        gpu_ids=self.gpu_ids,
                                        ds_n=opt.downsampling_0_n,
                                        ds_mult=opt.downsampling_0_mult,
                                        ds_post=opt.dspost_G)
        self.netGg = XInetworks.define_G(opt.input_nc, opt.output_nc,
                                         opt.ngf, opt.which_model_netG,
                                         opt.norm, opt.use_dropout,
                                         gpu_ids=self.gpu_ids,
                                         ds_n=opt.downsampling_0_n,
                                         ds_mult=opt.downsampling_0_mult,
                                         ds_post=opt.dspost_G)
        disc_ch = opt.input_nc
            
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.conditional:
                if opt.which_model_preNet != 'none':
                    self.preNet_A = XInetworks.define_preNet(disc_ch + disc_ch, disc_ch + disc_ch,
                                                             which_model_preNet=opt.which_model_preNet, norm=opt.norm, gpu_ids=self.gpu_ids)
 #                   self.preNet_Ag = networks.define_preNet(disc_ch + disc_ch, disc_ch + disc_ch,
 # which_model_preNet=opt.which_model_preNet, norm=opt.norm,gpu_ids=self.gpu_ids)
                nif = disc_ch+disc_ch

                netD_norm = opt.norm
#                netDg_norm = opt.norm

                self.netDg = XInetworks.define_D(nif, opt.ndf, opt.which_model_netD, opt.n_layers_D, netD_norm,
                                                 use_sigmoid, gpu_ids=self.gpu_ids)
#                self.netDg = networks.define_D(nif, opt.ndf, opt.which_model_netD, opt.n_layers_D, netD_norm,
# use_sigmoid, gpu_ids=self.gpu_ids)

            else:
                self.netDg = XInetworks.define_D(disc_ch, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,
                                                 use_sigmoid, gpu_ids=self.gpu_ids)
#                self.netDg = networks.define_D(disc_ch, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,
# use_sigmoid, gpu_ids=self.gpu_ids)
        latest = opt.continue_latest
        if not self.isTrain or (opt.which_epoch > 0):
            if self.opt.conv3d:
                 self.load_network(self.netG_3d, 'G_3d', latest=latest)
                 self.load_network(self.netGg_3d, 'Gg_3d', latest=latest)
            self.load_network(self.netG, 'G', latest=latest)
            self.load_network(self.netGg, 'Gg', latest=latest)
            if self.isTrain:
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', latest=latest)
                    # self.load_network(self.preNet_Ag, 'PRE_Ag', opt.which_epoch)
                self.load_network(self.netDg, 'Dg', latest=latest)
                # self.load_network(self.netDg, 'Dg', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = XInetworks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()


            # initialize optimizers
            if self.opt.conv3d:
                 self.optimizer_G_3d = torch.optim.Adam(self.netG_3d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                 self.optimizer_Gg_3d = torch.optim.Adam(self.netGg_3d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gg = torch.optim.Adam(self.netGg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(self.preNet_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#                self.optimizer_preAg = torch.optim.Adam(self.preNet_Ag.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dg = torch.optim.Adam(self.netDg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            if self.opt.conv3d:
                XInetworks.print_network(self.netG_3d)
                XInetworks.print_network(self.netGg_3d)
            XInetworks.print_network(self.netG)
            XInetworks.print_network(self.netGg)
            if opt.which_model_preNet != 'none':
                XInetworks.print_network(self.preNet_A)
#                networks.print_network(self.preNet_Ag)
#             XInetworks.print_network(self.netD)
            XInetworks.print_network(self.netDg)
            print('-----------------------------------------------')
            if not self.isTrain or (opt.which_epoch > 0):
                print('LOADING from %s' % self.epoch_str)

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']        
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
            self.fake_B_indep = self.netGg_3d.forward(self.fake_B.unsqueeze(2))
            self.refake_A = self.netGg.forward(self.fake_B_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)
            self.refake_A = self.netGg.forward(self.fake_B)

        self.real_B = Variable(self.input_B)
        real_B = util.tensor2im(self.real_B.data)
        real_A = util.tensor2im(self.real_A.data)
    
    def add_noise_disc(self,real):
        #add noise to the discriminator target labels
        #real: True/False? 
        if self.opt.noisy_disc:
            rand_lbl = random.random()
            if rand_lbl<0.6:
                label = (not real)
            else:
                label = (real)
        else:  
            label = (real)
        return label
    
    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
            self.fake_B_indep = self.netGg_3d.forward(self.fake_B.unsqueeze(2))
            self.refake_A = self.netGg.forward(self.fake_B_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)
            self.refake_A = self.netGg.forward(self.fake_B)
            
        self.real_B = Variable(self.input_B, volatile=True)

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
                            
        else:
            self.pred_fake = self.netD.forward(self.fake_B.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, label_fake)

        # Real
        label_real = self.add_noise_disc(True)
        if self.opt.conditional:
            real_AB = torch.cat((self.real_A_reshaped, self.real_B_reshaped), 1)#.detach()
            self.pred_real_patch = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real_patch, label_real)

            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_A_real = self.preNet_A.forward(real_AB)
                self.pred_real = self.netD.forward(transformed_A_real)
                self.loss_D_real += self.criterionGAN(self.pred_real, label_real)
                            
        else:
            self.pred_real = self.netD.forward(self.real_B)            
            self.loss_D_real = self.criterionGAN(self.pred_real, label_real)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

# to be edited
    def backward_Dg(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)

        b, c, m, n = self.fake_B.size()
        rgb = 3 if self.opt.rgb else 1

        self.real_A_reshaped = self.real_A
        self.real_B_reshaped = self.real_B
        self.refake_A_reshaped = self.refake_A

        if self.opt.conditional:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A_reshaped, self.refake_A_reshaped), 1))
            self.pred_fake_patch = self.netDg.forward(fake_AB.detach())
            self.loss_Dg_fake = self.criterionGAN(self.pred_fake_patch, label_fake)
            if self.opt.which_model_preNet != 'none':
                # transform the input
                transformed_AB = self.preNet_A.forward(fake_AB.detach())
                self.pred_fake = self.netDg.forward(transformed_AB)
                self.loss_Dg_fake += self.criterionGAN(self.pred_fake, label_fake)

        else:
            self.pred_fake = self.netDg.forward(self.refake_A.detach())
            self.loss_Dg_fake = self.criterionGAN(self.pred_fake, label_fake)

        # Real
        label_real = self.add_noise_disc(True)
        if self.opt.conditional:
            real_AB = torch.cat((self.real_A_reshaped, self.real_B_reshaped), 1)  # .detach()
            self.pred_real_patch = self.netDg.forward(real_AB)
            self.loss_Dg_real = self.criterionGAN(self.pred_real_patch, label_real)

            if self.opt.which_model_preNet != 'none':
                # transform the input
                transformed_A_real = self.preNet_A.forward(real_AB)
                self.pred_real = self.netDg.forward(transformed_A_real)
                self.loss_Dg_real += self.criterionGAN(self.pred_real, label_real)

        else:
            self.pred_real = self.netDg.forward(self.real_B)
            self.loss_Dg_real = self.criterionGAN(self.pred_real, label_real)

        # Combined loss
        self.loss_Dg = (self.loss_Dg_fake + self.loss_Dg_real) * 0.5

        self.loss_Dg.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.conditional:
            #PATCH GAN
            fake_AB = (torch.cat((self.real_B_reshaped, self.refake_A_reshaped), 1))
            pred_fake_patch = self.netDg.forward(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
            if self.opt.which_model_preNet != 'none':
                #global disc
                transformed_A = self.preNet_A.forward(fake_AB)
                pred_fake = self.netDg.forward(transformed_A)
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
        else:
            pred_fake = self.netDg.forward(self.refake_A)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_Gg_L1 = self.criterionL1(self.refake_A, self.real_B) * self.opt.lambda_A
        # self.loss_G_cycleGAN = self.criterionL1(self.real_A, self.refake_A) * self.opt.lambda_A
#        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_Gg_L1 + self.loss_G_cycleGAN
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * 1/3 + self.loss_Gg_L1 * 2/3

        self.loss_G.backward()

#    def backward_Gg(self):
#        # First, G(A) should fake the discriminator
#        if self.opt.conditional:
#            # PATCH GAN
#            fake_AB = (torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
#            pred_fake_patch = self.netD.forward(fake_AB)
#            self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
#            if self.opt.which_model_preNet != 'none':
#                # global disc
#                transformed_A = self.preNet_A.forward(fake_AB)
#                pred_fake = self.netD.forward(transformed_A)
#                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
#       else:
#            pred_fake = self.netD.forward(self.fake_B)
#            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
#        self.loss_Gg_L1 = self.criterionL1(self.refake_A, self.real_B) * self.opt.lambda_A
#        self.loss_Gg = self.loss_Gg_L1
#        self.loss_Gg.backward()

    def optimize_parameters(self):
        self.forward()

        # self.optimizer_D.zero_grad()
        self.optimizer_Dg.zero_grad()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.zero_grad()
#            self.optimizer_preAg.zero_grad()
#         self.backward_D()
        self.backward_Dg()
        # self.optimizer_D.step()
        self.optimizer_Dg.step()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.step()
#            self.optimizer_preAg.step()

        self.optimizer_Gg.zero_grad()
        self.optimizer_G.zero_grad()
        if self.opt.conv3d:
            self.optimizer_Gg_3d.zero_grad()
            self.optimizer_G_3d.zero_grad()

        self.backward_G()
#        self.backward_Gg()
        self.optimizer_Gg.step()
        if self.opt.conv3d:
            self.optimizer_Gg_3d.step()
        self.optimizer_G.step()
        if self.opt.conv3d:
            self.optimizer_G_3d.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('Gg_L1', self.loss_Gg_L1.data),
                            ('Dg_real', self.loss_Dg_real.data),
                            ('Dg_fake', self.loss_Dg_fake.data)
                            # ('cycle_GAN', self.loss_G_cycleGAN)
        ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        refake_A = util.tensor2im(self.refake_A.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('refake_A', refake_A)])

    def save(self, latest=False):
        if self.opt.conv3d:
            self.save_network(self.netG_3d, 'G_3d',
                              gpu_ids=self.gpu_ids, latest=latest)
            self.save_network(self.netGg_3d, 'Gg_3d', gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netG, 'G', gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netGg, 'Gg', gpu_ids=self.gpu_ids, latest=latest)
        # self.save_network(self.netD, 'D', gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netDg, 'Dg', gpu_ids=self.gpu_ids, latest=latest)
        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A', gpu_ids=self.gpu_ids, latest=latest)
           # self.save_network(self.preNet_A, 'PRE_A', gpu_ids=self.gpu_ids, latest=latest)

    def set_learning_rate(self, epoch):
        lr = self.opt.lr / self.batch_skip
        if epoch > self.opt.niter:
            lr = lr * (self.opt.niter + self.opt.niter_decay -
                       epoch) / self.opt.niter_decay
        for param_group in self.optimizer_Dg.param_groups:
            param_group['lr'] = lr
        if self.opt.which_model_preNet != 'none':
            for param_group in self.optimizer_preA.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_Gg.param_groups:
            param_group['lr'] = lr
        if self.opt.conv3d:
            for param_group in self.optimizer_G_3d.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_Gg_3d.param_groups:
                param_group['lr'] = lr
        print('setting learning rate: %f at epoch %d' % (lr, epoch))
