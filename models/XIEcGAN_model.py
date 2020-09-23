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


class EcGANModel(BaseModel):
    def name(self):
        return 'EcGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.output_str_indices = str2index(self.opt.str_output,
                                            self.opt.charset)
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize * 3)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize * 3)

        # load/define networks
        if self.opt.conv3d:
            self.netG_3d = XInetworks.define_G_3d(opt.input_nc,
                                                  opt.input_nc,
                                                  norm=opt.norm,
                                                  groups=opt.grps,
                                                  gpu_ids=self.gpu_ids)

        self.netG = XInetworks.define_G(opt.input_nc, opt.output_nc,
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
                    self.preNet_A = XInetworks.define_preNet(
                        disc_ch + disc_ch, disc_ch + disc_ch,
                        which_model_preNet=opt.which_model_preNet,
                        norm=opt.norm, gpu_ids=self.gpu_ids)
                nif = disc_ch + disc_ch
                netD_norm = opt.norm
                self.netD = XInetworks.define_D(nif, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D,
                                                netD_norm,
                                                use_sigmoid,
                                                gpu_ids=self.gpu_ids)
            else:
                self.netD = XInetworks.define_D(disc_ch, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D,
                                                opt.norm, use_sigmoid,
                                                gpu_ids=self.gpu_ids)
        latest = opt.continue_latest
        if not self.isTrain or (opt.which_epoch > 0):
            if self.opt.conv3d:
                self.load_network(self.netG_3d, 'G_3d', latest=latest)
            self.load_network(self.netG, 'G', latest=latest)
            if self.isTrain:
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A',
                                      latest=latest)
                self.load_network(self.netD, 'D', latest=latest)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = XInetworks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            if opt.diff_loss0 == 'L1':
                self.criterionDiff0 = self.criterionL1
            elif opt.diff_loss0 == 'MSE':
                self.criterionDiff0 = self.criterionMSE

            # initialize optimizers
            if self.opt.conv3d:
                self.optimizer_G_3d = torch.optim.Adam(
                    self.netG_3d.parameters(),
                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(
                    self.preNet_A.parameters(),
                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999))

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
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        # print('REAL_A size is')
        # print(self.real_A.size())
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(
                self.real_A.unsqueeze(2))
            #            print('REAL_A_INDEP size is')
            #            print(self.real_A_indep.size())
            self.fake_B = self.netG.forward(
                self.real_A_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)

        #        print('FAKE_B size is')
        #        print(self.fake_B.size())

        b, c, m, n = self.fake_B.size()
        self.real_AA = self.real_A.narrow(3, m, m)
        self.fake_BB = self.fake_B.narrow(3, m, m)
        self.real_B = Variable(self.input_B)
        self.real_BB = self.real_B.narrow(3, m, m)
        #        print('REAL_B size is')
        #        print(self.real_B.size())
        real_B = util.tensor2im(self.real_B.data)
        real_A = util.tensor2im(self.real_A.data)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(
                self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(
                self.real_A_indep.squeeze(2))

        else:
            self.fake_B = self.netG.forward(self.real_A)

        b, c, m, n = self.fake_B.size()
        self.real_AA = self.real_A.narrow(3, m, m)
        self.fake_BB = self.fake_B.narrow(3, m, m)
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_BB = self.real_B.narrow(3, m, m)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)

        b, c, m, n = self.fake_B.size()
        rgb = 3 if self.opt.rgb else 1

        self.fake_B_reshaped = self.fake_B
        self.real_A_reshaped = self.real_A
        self.real_B_reshaped = self.real_B
        self.fake_BB_reshaped = self.fake_BB
        self.real_AA_reshaped = self.real_AA
        self.real_BB_reshaped = self.real_BB

        if self.opt.conditional:
            fake_AB = self.fake_AB_pool.query(torch.cat(
                (self.real_A_reshaped, self.fake_B_reshaped), 1))
            self.pred_fake_patch = self.netD.forward(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake_patch,
                                                 label_fake)
            if self.opt.which_model_preNet != 'none':
                # transform the input
                transformed_AB = self.preNet_A.forward(
                    fake_AB.detach())
                self.pred_fake = self.netD.forward(transformed_AB)
                self.loss_D_fake += self.criterionGAN(self.pred_fake,
                                                      label_fake)
        else:
            self.pred_fake = self.netD.forward(
                self.fake_B_reshaped.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake,
                                                 label_fake)

        # Real
        label_real = self.add_noise_disc(True)
        if self.opt.conditional:
            real_AB = torch.cat(
                (self.real_A_reshaped, self.real_B_reshaped),
                1)  # .detach()
            self.pred_real_patch = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real_patch,
                                                 label_real)

            if self.opt.which_model_preNet != 'none':
                # transform the input
                transformed_A_real = self.preNet_A.forward(real_AB)
                self.pred_real = self.netD.forward(transformed_A_real)
                self.loss_D_real += self.criterionGAN(self.pred_real,
                                                      label_real)
        else:
            self.pred_real = self.netD.forward(self.real_B)
            self.loss_D_real = self.criterionGAN(self.pred_real,
                                                 label_real)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.conditional:
            # PATCH GAN
            fake_AB = (torch.cat(
                (self.real_A_reshaped, self.fake_B_reshaped), 1))
            pred_fake_patch = self.netD.forward(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
            if self.opt.which_model_preNet != 'none':
                # global disc
                transformed_A = self.preNet_A.forward(fake_AB)
                pred_fake = self.netD.forward(transformed_A)
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
        else:
            pred_fake = self.netD.forward(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_BB, self.real_BB) \
                         * self.opt.lambda_A
        self.loss_G_MSE = self.criterionMSE(self.fake_BB,
                                            self.real_BB) \
                          * self.opt.lambda_A

        self.loss_G = self.criterionDiff0(self.fake_BB,
                                          self.real_BB) \
                      * self.opt.lambda_A
        # self.loss_G = self.loss_G_L1 * self.loss_G_MSE / 20
        self.loss_G = self.loss_G + self.loss_G_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # for name, param in self.netG.named_parameters():
        #     if param.requires_grad:
        #         print(name, sum(sum(sum(param.data))))
        #     break

        if (self.skip_batch + 1) % self.batch_skip == 0:
            self.optimizer_D.zero_grad()
            if self.opt.which_model_preNet != 'none':
                self.optimizer_preA.zero_grad()

        self.backward_D()

        if (self.skip_batch + 1) % self.batch_skip == 0:
            self.optimizer_D.step()
            if self.opt.which_model_preNet != 'none':
                self.optimizer_preA.step()

        if (self.skip_batch + 1) % self.batch_skip == 0:
            self.optimizer_G.zero_grad()
            if self.opt.conv3d:
                self.optimizer_G_3d.zero_grad()

        self.backward_G()

        if (self.skip_batch + 1) % self.batch_skip == 0:
            self.optimizer_G.step()
            if self.opt.conv3d:
                self.optimizer_G_3d.step()

        self.skip_batch = (self.skip_batch + 1) % self.batch_skip

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('G_MSE', self.loss_G_MSE.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_fake', self.loss_D_fake.data)
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_AA = util.tensor2im(self.real_AA.data)
        fake_BB = util.tensor2im(self.fake_BB.data)
        real_BB = util.tensor2im(self.real_BB.data)
        if self.output_str_indices:
            output_str_indices = torch.LongTensor(
                self.output_str_indices)
            output_str_data = self.fake_BB.data[:, output_str_indices,
                              :, :]
            real_str_data = self.real_BB.data[:, output_str_indices,
                            :, :]
            img_output_str = util.tensor2im(output_str_data)
            img_real_str = util.tensor2im(real_str_data)
            list = [('Ereal_A', real_A), ('Ereal_B', real_B),
                    ('Efake_B', fake_B),  ('real_A', real_AA),
                    ('real_B', real_BB), ('fake_B', fake_BB),
                    ('fake_STR', img_output_str),
                    ('real_STR', img_real_str)]
        else:
            list = [('Ereal_A', real_A), ('Ereal_B', real_B),
                    ('Efake_B', fake_B),  ('real_A', real_AA),
                    ('real_B', real_BB), ('fake_B', fake_BB)]
        if self.opt.str_input:
            list = [('real_A', real_AA), ('real_B', real_BB),
                    ('fake_B', fake_BB)]
        return OrderedDict(list)

    def save(self, latest=False):
        if self.opt.conv3d:
            self.save_network(self.netG_3d, 'G_3d',
                              gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netG, 'G', gpu_ids=self.gpu_ids,
                          latest=latest)
        self.save_network(self.netD, 'D', gpu_ids=self.gpu_ids,
                          latest=latest)
        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A',
                              gpu_ids=self.gpu_ids, latest=latest)
