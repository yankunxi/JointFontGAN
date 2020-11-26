#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================

import torch
from collections import OrderedDict
from torch.autograd import Variable
import JointFontGAN.util.XIutil as util
from JointFontGAN.util.image_pool import ImagePool
from JointFontGAN.util.indexing import str2index
from .XIbase_model import BaseModel
from . import XInetworks
import random



class cGANModel(BaseModel):
    def name(self):
        return 'cGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        # self.input_RB = self.Tensor(opt.batchSize, opt.input_nc,
        #                             opt.fineSize, opt.fineSize)

        self.output_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.output_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        self.str_image = self.Tensor(opt.batchSize,
                                     len(opt.str_output),
                                     opt.fineSize, opt.fineSize)

        # load/define networks
        if self.opt.conv3d:
            self.netG_3d = XInetworks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)

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
                    self.preNet_A = XInetworks.define_preNet(disc_ch + disc_ch, disc_ch + disc_ch, which_model_preNet=opt.which_model_preNet, norm=opt.norm, gpu_ids=self.gpu_ids)
                nif = disc_ch+disc_ch
                netD_norm = opt.norm
                self.netD = XInetworks.define_D(nif, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, netD_norm, use_sigmoid, gpu_ids=self.gpu_ids)
            else:
                self.netD = XInetworks.define_D(disc_ch, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, gpu_ids=self.gpu_ids)
        latest = opt.continue_latest
        if not self.isTrain or (opt.which_epoch > 0):
            if self.opt.conv3d:
                self.load_network(self.netG_3d, 'G_3d', latest=latest)
            self.load_network(self.netG, 'G', latest=latest)
            if self.isTrain:
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', latest=latest)
                self.load_network(self.netD, 'D', latest=latest)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = XInetworks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.diff_loss0 == 'L1':
                self.criterionDiff0 = torch.nn.L1Loss()
            elif opt.diff_loss0 == 'MSE':
                self.criterionDiff0 = torch.nn.MSELoss()

            # initialize optimizers
            if self.opt.conv3d:
                 self.optimizer_G_3d = torch.optim.Adam(self.netG_3d.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(self.preNet_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
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

        if self.opt.skmode == 0:
            input_A = input['Ask0']
            input_B = input['Bsk0']
        elif self.opt.skmode == 1:
            input_A = input['Ask1']
            input_B = input['Bsk1']
        else:
            input_A = input['A']
            input_B = input['B']
            # input_RB = input['RB']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        output_A = input_A
        output_B = input_B

        # self.input_RB.resize_(input_RB.size()).copy_(input_RB)

        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = self.input_A.detach()
        self.real_B = self.output_B.detach()
        # self.redbox = self.input_RB.detach()
        # print('REAL_A size is')
        # print(self.real_A.size())
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)
    
    # no backprop gradients
    def test(self):
        # for m in self.netG.modules():
        #     # if isinstance(m, torch.nn.BatchNorm2d) | isinstance(m, torch.nn.Dropout):
        #     m.eval()
        # for m in self.netG_3d.modules():
        #     # if isinstance(m, torch.nn.BatchNorm3d):
        #     m.eval()

        self.real_A = self.input_A.detach()
        self.real_B = self.output_B.detach()
        # print('REAL_A size is')
        # print(self.real_A.size())
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)

        # self.loss_G_L1 = self.criterionDiff0(self.fake_B, self.real_B)
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
                            
        else:
            self.pred_fake = self.netD.forward(self.fake_B.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, label_fake)

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
                            
        else:
            self.pred_real = self.netD.forward(self.real_B)            
            self.loss_D_real = self.criterionGAN(self.pred_real, label_real)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
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
        else:
            pred_fake = self.netD.forward(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G_diff = self.criterionDiff0(self.fake_B,
                                               self.real_B) * \
                           self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_diff

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if self.skip_batch % self.batch_skip == 0:
            self.optimizer_D.zero_grad()
            if self.opt.which_model_preNet != 'none':
                self.optimizer_preA.zero_grad()

        self.backward_D()

        if (self.skip_batch + 1) % self.batch_skip == 0:
            self.optimizer_D.step()
            if self.opt.which_model_preNet != 'none':
                self.optimizer_preA.step()

        if self.skip_batch % self.batch_skip == 0:
            self.optimizer_G.zero_grad()
            if self.opt.conv3d:
                self.optimizer_G_3d.zero_grad()

        self.backward_G()

        if (self.skip_batch + 1) % self.batch_skip == 0:
            self.optimizer_G.step()
            if self.opt.conv3d:
                self.optimizer_G_3d.step()


    def get_current_errors(self):
        errors = [('G_GAN', self.loss_G_GAN.data)]
        if self.opt.diff_loss0 == 'L1':
            errors = errors + [('G_L1', self.loss_G_diff.data)]
        elif self.opt.diff_loss0 == 'MSE':
            errors = errors + [('G_MSE', self.loss_G_diff.data)]
        errors = errors + [('D_real', self.loss_D_real.data),
                           ('D_fake', self.loss_D_fake.data)]
        return OrderedDict(errors)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        # redbox = util.tensor2im(self.redbox.data)

        output_str_indices = str2index(self.opt.str_output,
                                       self.opt.charset)
        if output_str_indices:
            output_str_indices = torch.LongTensor(output_str_indices)
            output_str_data = self.fake_B.data[:, output_str_indices, :, :]
            real_str_data = self.real_B.data[:, output_str_indices, :, :]
            img_output_str = util.tensor2im(output_str_data)
            img_real_str = util.tensor2im(real_str_data)
            return OrderedDict([('real_A', real_A), 
                                ('real_B', real_B),
                                ('fake_B', fake_B),
                                ('fake_STR', img_output_str),
                                ('real_STR', img_real_str)])
        else:
            return OrderedDict([('real_A', real_A),
                                ('real_B', real_B),
                                ('fake_B', fake_B)
                                # , ('red_box', redbox)
                                ])

    def save(self, latest=False):
        if self.opt.conv3d:
             self.save_network(self.netG_3d, 'G_3d',
                               gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netG, 'G', gpu_ids=self.gpu_ids, latest=latest)
        self.save_network(self.netD, 'D', gpu_ids=self.gpu_ids, latest=latest)
        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A', gpu_ids=self.gpu_ids, latest=latest)

    def set_learning_rate(self, epoch):
        lr = self.opt.lr / self.batch_skip
        if epoch > self.opt.niter:
            lr = lr * (self.opt.niter + self.opt.niter_decay -
                       epoch) / self.opt.niter_decay
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if self.opt.which_model_preNet != 'none':
            for param_group in self.optimizer_preA.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.conv3d:
            for param_group in self.optimizer_G_3d.param_groups:
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
