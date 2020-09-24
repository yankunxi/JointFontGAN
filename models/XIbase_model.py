#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================

import os
import torch
import numpy as np


class BaseModel():
    def name(self):
        return 'BaseModel'

    # def __init__(self):

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.skmode = self.opt.skmode
        self.batch_skip = opt.batchSplit
        self.skip_batch = 0
        self.set_epoch(opt.which_epoch)
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.opt.use_auxiliary:
            self.save_dir = os.path.join(self.opt.auxiliary_root,
                                         self.opt.project_relative,
                                         self.opt.checkpoints_dir,
                                         self.opt.experiment_dir)
        else:
            self.save_dir = os.path.join(self.opt.everything_root,
                                         self.opt.project_relative,
                                         self.opt.checkpoints_dir,
                                         self.opt.experiment_dir)

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
        return torch.tensor(label, dtype=torch.bool).cuda()

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, gpu_ids,
                      latest=False, epoch='_'):
        if epoch == '_':
            epoch = self.epoch_str
        if latest:
            epoch = 'latest'
        save_filename = '%s_net_%s.pth' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    def save_parameter(self, parameter, parameter_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, parameter_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(parameter.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            parameter.cuda(device=gpu_ids[0])

    def load_combo_network(self, network1, network2, network_label, epoch_label,print_weights=False,ignore_BN=False):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        weights = torch.load(save_path)
        # print weights
        if ignore_BN:
            for key in weights.keys():
                if key.endswith('running_mean'):
                    weights[key].zero_()
                elif key.endswith('running_var'):
                    weights[key].fill_(1.0)
        if print_weights:
            for key in weights.keys():
                print(key, 'pretrained, mean,std:', torch.mean(weights[key]),torch.std(weights[key]))


        keys1 = network1.state_dict().keys()
        weights1={}
        for key in keys1:
            weights1[key] = weights[key]
        network1.load_state_dict(weights1)
        weights2={}

        keys2 = network2.state_dict().keys()
        keys2_in_weights = list(set(weights.keys())-set(keys1))
        keys1_last_lyr_number = max([int(key.split(".")[1]) for key in keys1])
        for old_key in keys2_in_weights:
            old_key_i = old_key.split(".")
            lyr_num = str(int(old_key_i[1])-keys1_last_lyr_number-1)
            old_key_p2 = old_key.split(''.join([old_key_i[0],'.',old_key_i[1]]))[1]
            new_key = ''.join([old_key_i[0],'.',lyr_num])
            new_key = ''.join([new_key,old_key_p2])
            weights2[new_key] = weights[old_key]
        
        network2.load_state_dict(weights2)

    # helper loading function that can be used by subclasses

    def load_network(self, network, network_label, latest=False,
                     print_weights=False, ignore_BN=False, epoch='_'):
        if epoch == '_':
            epoch = self.epoch_str
        if latest:
            epoch = 'latest'
        save_filename = '%s_net_%s.pth' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        weights = torch.load(save_path)
        # print weights
        if ignore_BN:
            for key in weights.keys():
                if key.endswith('running_mean'):
                    weights[key].zero_()
                elif key.endswith('running_var'):
                    weights[key].fill_(1.0)
        if print_weights:
            for key in weights.keys():
                print(key, 'pretrained, mean,std:', torch.mean(weights[key]),
                      torch.std(weights[key]))
        network.load_state_dict(weights)

    def load_parameter(self, parameter, parameter_label, epoch_label,
                       print_weights=False):
        save_filename = '%s_net_%s.pth' % (epoch_label, parameter_label)
        save_path = os.path.join(self.save_dir, save_filename)

        weights = torch.load(save_path)
        # print weights
        if print_weights:
            for key in weights.keys():
                print(key, 'pretrained, mean,std:', torch.mean(weights[key]),
                      torch.std(weights[key]))

        parameter.load_state_dict(weights)

    def next_epoch(self):
        self.set_epoch(self.epoch + 1)
        self.set_learning_rate(self.epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.parse_epoch()

    def set_learning_rate(self, epoch):
        lr = self.opt.lr/self.batch_skip
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

    def parse_epoch(self):
        if self.epoch > self.opt.niter:
            self.epoch_str = str(self.opt.niter) + '+' + str(
                self.epoch - self.opt.niter) + '@' + str(
                self.opt.niter_decay)
        else:
            self.epoch_str = str(self.epoch) + '+0'
        if self.epoch == 0:
            self.epoch_str1 = self.opt.which_epoch1
        else:
            self.epoch_str1 = self.opt.which_epoch1 + '=' + self.epoch_str


