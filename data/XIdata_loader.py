#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================

from JointFontGAN.data.XIimage_folder import ImageFolder
from JointFontGAN.data.XIbase_data_loader import BaseDataLoader
import random
import torch.utils.data
import torchvision.transforms as transforms
from builtins import object
import os
import numpy as np
from torch import LongTensor
import warnings
import pickle
from skimage.morphology import skeletonize
from skimage.util import invert


def normalize_stack(input, val=0.5):
    # normalize an tensor with arbitrary number of channels:
    # each channel with mean=std=val
    val=0.5
    len_ = input.size(0)
    mean = (val,)*len_
    std = (val,)*len_
    t_normal_stack = transforms.Compose([
        transforms.Normalize(mean, std)])
    return t_normal_stack(input)


def CreateDataLoader(opt):
    data_loader = None
    # if opt.stack:
    #     data_loader = StackDataLoader()
    # elif opt.partial:
    #     data_loader = PartialDataLoader()
    if opt.data_loader == 'base':
        data_loader = DataLoader()
    elif opt.data_loader == 'stack':
        data_loader = StackDataLoader()
    elif opt.data_loader == 'Estack':
        data_loader = EStackDataLoader()
    elif opt.data_loader == 'partial':
        data_loader = PartialDataLoader()
    elif opt.data_loader == 'extended_half':
        data_loader = ExHalfDataLoader()
    elif opt.data_loader == 'extended_half_t':
        data_loader = ExHalfTDataLoader()
    elif opt.data_loader == 'extended_half_t_sk':
        data_loader = ExHalfTskDataLoader()
    elif opt.data_loader == 'skeleton':
        data_loader = skDataLoader()
    elif opt.data_loader == 'EHskeleton':
        data_loader = EHskDataLoader()
    data_loader.initialize(opt)

    print(data_loader.name())
    return data_loader


class FlatData(object):
    def __init__(self, data_loader, data_loader_base, fineSize, max_dataset_size, rgb, dict_test={},base_font=False, blanks=0.7):
        self.data_loader = data_loader
        self.data_loader_base = data_loader_base
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.blanks = blanks
        self.data_loader_base_iter = iter(self.data_loader_base)
        self.A_base,self.A_base_paths = next(self.data_loader_base_iter)
        self.base_font = base_font
        self.random_dict=dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total/2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = torch.mean(AB, dim=1)  # only one of the RGB channels
        A = A[:,None,:,:]  # (m,1,64,64*26)
        B = torch.mean(AB, dim=1)  # only one of the RGB channels
        B = B[:,None,:,:]  # (m,1,64,64*26)
        n_rgb = 3 if self.rgb else 1
        target_size = A.size(2)
        AA = A.clone()
        if self.blanks != 0:
            # randomly remove some of the glyphs
            if not self.dict:
                blank_ind = np.random.permutation(A.size(3)/target_size)[0:int(self.blanks*A.size(3)/target_size)]
            else:
                file_name = map(lambda x:x.split("/")[-1],AB_paths)
                blank_ind = self.random_dict[file_name][0:int(self.blanks*A.size(3)/target_size)]

            blank_ind = np.tile(range(target_size), len(blank_ind)) + np.repeat(blank_ind*target_size,target_size)
            AA.index_fill_(3,LongTensor(list(blank_ind)),1)
            # t_topil = transforms.Compose([
            #     transforms.ToPILImage()])

            # AA_ = t_topil(AA[0,0,:,:].unsqueeze_(0))
            # misc.imsave('./AA_0.png',AA_)

        return {'A': AA, 'A_paths': AB_paths, 'B':B, 'B_paths':AB_paths}


# generate alphabet
class Data(object):
    def __init__(self, str_indices, data_loader, fineSize,
                 max_dataset_size, rgb, dict_test={}, blanks=0.7):
        self.str_indices = str_indices
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.n_rgb = 3 if self.rgb else 1
        self.blanks = blanks
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset: w_offset + self.fineSize]
        self.ab_size = int(A.size(1) / self.n_rgb)

        remain_ind = list(set(self.str_indices) - {-1})
        if remain_ind:
            self.remainnum = len(remain_ind)
            self.blanknum = self.ab_size - self.remainnum
            blank_ind = list(set(range(self.ab_size)) - set(remain_ind))
        else:
            self.blanknum = int(self.blanks * self.ab_size)
            self.remainnum = self.ab_size - self.blanknum
            # randomly remove some of the glyphs in input
            if not self.dict:
                blank_ind = np.random.permutation(
                    range(self.ab_size))[0: self.blanknum]
                # print(blank_ind)
            else:
                file_name = list(map(lambda x: x.split("/")[-1],
                                     AB_paths))
                if len(file_name) > 1:
                    raise Exception('batch size should be 1')
                file_name = file_name[0]
                blank_ind = self.random_dict[file_name][0:self.blanknum]
        rgb_inds = np.tile(range(self.n_rgb), self.blanknum)
        blank_ind = np.repeat(blank_ind, self.n_rgb)
        blank_ind = blank_ind * self.n_rgb + rgb_inds
        AA = A.clone()
        AA.index_fill_(1, LongTensor(list(blank_ind)), 1)
        # A_ab = A.clone()
        #
        # A_ab.index_fill_(1, 1, LongTensor(list(blank_ind)),
        #                  transforms.ToTensor(Image.fromarray()))
        return {'A': AA, 'A_paths': AB_paths, 'B': B,
                'B_paths': AB_paths,
                # 'RB': A_ab,
                'indices': remain_ind}


# generate alphabet with skeletons
class skData(object):
    def __init__(self, str_indices, data_loader, fineSize,
                 max_dataset_size, rgb, dict_test={}, blanks=0.7,
                 skmode=1):
        self.skmode = skmode
        self.str_indices = str_indices
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.n_rgb = 3 if self.rgb else 1
        self.blanks = blanks
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        self.ab_size = int(A.size(1) / self.n_rgb)

        remain_ind = list(set(self.str_indices) - {-1})
        if remain_ind:
            self.remainnum = len(remain_ind)
            self.blanknum = self.ab_size - self.remainnum
            blank_ind = list(set(range(self.ab_size)) - set(remain_ind))
        else:
            self.blanknum = int(self.blanks * self.ab_size)
            self.remainnum = self.ab_size - self.blanknum
            # randomly remove some of the glyphs in input
            if not self.dict:
                blank_ind = np.random.permutation(
                    range(self.ab_size))[0: self.blanknum]
                # print(blank_ind)
            else:
                file_name = list(map(lambda x: x.split("/")[-1],
                                     AB_paths))
                if len(file_name) > 1:
                    raise Exception('batch size should be 1')
                file_name = file_name[0]
                blank_ind = self.random_dict[file_name][0:self.blanknum]
        rgb_inds = np.tile(range(self.n_rgb), self.blanknum)
        blank_ind = np.repeat(blank_ind, self.n_rgb)
        blank_ind = blank_ind * self.n_rgb + rgb_inds
        AA = A.clone()
        B = A.clone()
        A.cpu()

        if not self.skmode == 1:
            Bsk0 = A.numpy()
            for i in range(A.size(0)):
                for j in range(A.size(1)):
                    Bsk0[i, j, :, :] = skeletonize(Bsk0[i, j, :,
                                                   :] > 0)
                    Bsk0[i, j, :, :] = invert(Bsk0[i, j, :, :])
            Bsk0 = LongTensor(Bsk0)
            Ask0 = Bsk0.clone()
            Ask0.index_fill_(1, LongTensor(list(blank_ind)), 1)
        if not self.skmode == 0:
            Bsk1 = A.numpy()
            for i in range(A.size(0)):
                for j in range(A.size(1)):
                    Bsk1[i, j, :, :] = invert(Bsk1[i, j, :, :])
                    Bsk1[i, j, :, :] = skeletonize(Bsk1[i, j,:,:] > 0)
                    Bsk1[i, j, :, :] = invert(Bsk1[i, j, :, :])
            Bsk1 = LongTensor(Bsk1)
            Ask1 = Bsk1.clone()
            Ask1.index_fill_(1, LongTensor(list(blank_ind)), 1)
        AA.index_fill_(1, LongTensor(list(blank_ind)), 1)

        if self.skmode == 0:
            return {'A': AA, 'A_paths': AB_paths, 'B': B,
                    'B_paths': AB_paths, 'Ask0': Ask0, 'Bsk0': Bsk0, 'indices': remain_ind}
        if self.skmode == 1:
            return {'A': AA, 'A_paths': AB_paths, 'B': B,
                    'B_paths': AB_paths, 'Ask1': Ask1, 'Bsk1': Bsk1, 'indices': remain_ind}
        if self.skmode == 2:
            return {'A': AA, 'A_paths': AB_paths, 'B': B,
                    'B_paths': AB_paths, 'Ask0': Ask0, 'Bsk0': Bsk0,
                    'Ask1': Ask1, 'Bsk1': Bsk1, 'indices': remain_ind}


# generate randomly extended alphabet with skeletons
class EHskData(object):
    def __init__(self, str_indices, data_loader, fineSize,
                 max_dataset_size, rgb, dict_test={}, blanks=0.7,
                 skmode=1):
        self.skmode = skmode
        self.str_indices = str_indices
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.n_rgb = 3 if self.rgb else 1
        self.blanks = blanks
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        self.ab_size = int(A.size(1) / self.n_rgb)

        remain_inds = list(set(self.str_indices) - {-1})
        if remain_inds:
            self.remainnum = len(remain_inds)
            self.blanknum = self.ab_size - self.remainnum
            blank_inds = list(set(range(self.ab_size)) - set(
                remain_inds))
        else:
            self.blanknum = int(self.blanks * self.ab_size)
            self.remainnum = self.ab_size - self.blanknum
            # randomly remove some of the glyphs in input
            if not self.dict:
                blank_inds = np.random.permutation(
                    range(self.ab_size))[0: self.blanknum]
                # print(blank_ind)
            else:
                file_name = list(map(lambda x: x.split("/")[-1],
                                     AB_paths))
                if len(file_name) > 1:
                    raise Exception('batch size should be 1')
                file_name = file_name[0]
                blank_inds = self.random_dict[file_name][
                             0:self.blanknum]
            remain_inds = list(set(range(self.ab_size)) - set(
                blank_inds))
        rgb_inds = np.tile(range(self.n_rgb), self.blanknum)
        blank_inds = np.repeat(blank_inds, self.n_rgb)
        blank_inds = blank_inds * self.n_rgb + rgb_inds
        # if not self.dict:
        L = A[:, [remain_inds[random.randint(0, self.remainnum -
                                             1)] for _ in range(A.size(1))], :, :]
        R = A[:, [remain_inds[random.randint(0, self.remainnum -
                                             1)] for _ in range(A.size(1))], :, :]
        # else:
        #     L = A[:, [remain_inds[((self.random_dict[file_name][
        #                                 (i * 9 + 7) % A.size(1)]
        #                             * 9 + 7) % self.remainnum)]
        #               for i in range(A.size(1))], :, :]
        #     R = A[:, [remain_inds[((self.random_dict[file_name][
        #                                 (i * 9 - 7) % A.size(1)]
        #                             * 9 + 7) % self.remainnum)]
        #               for i in range(A.size(1))], :, :]
        AA = torch.cat((L, A, R), 3)
        AAA = AA.clone()
        BB = torch.cat((L, B, R), 3)
        AA.cpu()

        if not self.skmode == 1:
            Bsk0 = AA.numpy()
            for i in range(AA.size(0)):
                for j in range(AA.size(1)):
                    Bsk0[i, j, :, :] = skeletonize(Bsk0[i, j, :,
                                                   :] > 0)
                    Bsk0[i, j, :, :] = invert(Bsk0[i, j, :, :])
            Bsk0 = LongTensor(Bsk0)
            Ask0 = Bsk0.clone()
            Ask0.index_fill_(1, LongTensor(list(blank_inds)), 1)
        if not self.skmode == 0:
            Bsk1 = AA.numpy()
            for i in range(AA.size(0)):
                for j in range(AA.size(1)):
                    Bsk1[i, j, :, :] = invert(Bsk1[i, j, :, :])
                    Bsk1[i, j, :, :] = skeletonize(Bsk1[i, j,:,:] > 0)
                    Bsk1[i, j, :, :] = invert(Bsk1[i, j, :, :])
            Bsk1 = LongTensor(Bsk1)
            Ask1 = Bsk1.clone()
            Ask1.index_fill_(1, LongTensor(list(blank_inds)), 1)

        AAA.index_fill_(1, LongTensor(list(blank_inds)), 1)

        if self.skmode == 0:
            return {'A': AAA, 'A_paths': AB_paths, 'B': BB,
                    'B_paths': AB_paths, 'Ask0': Ask0, 'Bsk0':
                        Bsk0, 'indices': remain_inds}
        if self.skmode == 1:
            return {'A': AAA, 'A_paths': AB_paths, 'B': BB,
                    'B_paths': AB_paths, 'Ask1': Ask1, 'Bsk1':
                        Bsk1, 'indices': remain_inds}
        if self.skmode == 2:
            return {'A': AAA, 'A_paths': AB_paths, 'B': BB,
                    'B_paths': AB_paths, 'Ask0': Ask0, 'Bsk0': Bsk0,
                    'Ask1': Ask1, 'Bsk1': Bsk1, 'indices':
                        remain_inds}


# generate randomly extended alphabet
class ExHalfData(object):
    def __init__(self, data_loader, fineSize, max_dataset_size, rgb,
                 dict_test={}, blanks=0.7):
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.n_rgb = 3 if self.rgb else 1
        self.blanks = blanks
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
            w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
            w_offset: w_offset + self.fineSize]
        self.ab_size = int(A.size(1) / self.n_rgb)
        self.blanknum = int(self.blanks * A.size(1) / self.n_rgb)
        self.remainnum = int(A.size(1) / self.n_rgb) - self.blanknum
        if self.blanks == 0:
            AA = A.clone()
        else:
            # randomly remove some of the glyphs in input
            if not self.dict:
                blank_inds = np.random.permutation(
                    range(self.ab_size))[0:self.blanknum]
            else:
                file_name = list(map(lambda x: x.split("/")[-1],
                                     AB_paths))
                if len(file_name) > 1:
                    raise Exception('batch size should be 1')
                file_name = file_name[0]
                blank_inds = self.random_dict[file_name][
                            0:self.blanknum]
            remain_inds = list(set(range(self.ab_size)) - set(
                blank_inds))
            # print(remain_inds)
            blank_inds = np.repeat(blank_inds, self.n_rgb)
            rgb_inds = np.tile(range(self.n_rgb), self.blanknum)
            blank_inds = blank_inds * self.n_rgb + rgb_inds
            if not self.dict:
                L = A[:, [remain_inds[random.randint(
                    0, self.remainnum - 1)] for _ in
                          range(A.size(1))], :, :]
                R = A[:, [remain_inds[random.randint(
                    0, self.remainnum - 1)] for _ in
                          range(A.size(1))], :, :]
            else:
                L = A[:, [remain_inds[((self.random_dict[file_name][
                                           (i * 9 + 7) % A.size(1)]
                                       * 9 + 7) % self.remainnum)]
                          for i in range(A.size(1))], :, :]
                R = A[:, [remain_inds[((self.random_dict[file_name][
                                            (i * 9 - 7) % A.size(1)]
                                        * 9 + 7) % self.remainnum)]
                          for i in range(A.size(1))], :, :]
            AA = torch.cat((L, A, R), 3)
            BB = torch.cat((L, B, R), 3)
            AA.index_fill_(1, LongTensor(list(blank_inds)), 1)
        return {'A': AA, 'A_paths': AB_paths, 'B': BB,
                'B_paths': AB_paths, 'indices': remain_inds}


# generate randomly extended alphabet along timestep
class ExHalfTData(object):
    def __init__(self, str_indices, data_loader, fineSize,
                 max_dataset_size, rgb,
                 dict_test={}, blanks=0.7):
        self.str_indices = str_indices
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.n_rgb = 3 if self.rgb else 1
        self.blanks = blanks
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
            w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
            w_offset: w_offset + self.fineSize]
        self.ab_size = int(A.size(1) / self.n_rgb)

        remain_inds = list(set(self.str_indices) - {-1})
        if remain_inds:
            self.str_num = len(self.str_indices)
            self.remainnum = len(remain_inds)
            self.blanknum = self.ab_size - self.remainnum
            blank_inds = list(
                set(range(self.ab_size)) - set(remain_inds))
        else:
            self.blanknum = int(self.blanks * A.size(1) / self.n_rgb)
            self.remainnum = int(
                A.size(1) / self.n_rgb) - self.blanknum
            self.str_num = self.remainnum
            # randomly remove some of the glyphs in input
            if not self.dict:
                blank_inds = np.random.permutation(
                    range(self.ab_size))[0:self.blanknum]
            else:
                file_name = list(map(lambda x: x.split("/")[-1],
                                  AB_paths))
                if len(file_name) > 1:
                    raise Exception('batch size should be 1')
                file_name = file_name[0]
                blank_inds = self.random_dict[file_name][
                            0:self.blanknum]
            remain_inds = list(set(range(self.ab_size))
                               - set(blank_inds))
            self.str_indices = remain_inds
            blank_ind = np.repeat(blank_inds, self.n_rgb)
        rgb_inds = np.tile(range(self.n_rgb), self.blanknum)
        blank_inds = blank_inds * self.n_rgb + rgb_inds

        AAA = A.unsqueeze(0)
        AAA = AAA.repeat([self.str_num, 1, 1, 1, 3])
        BBB = B.unsqueeze(0)
        BBB = BBB.repeat([self.str_num, 1, 1, 1, 3])
        for i in range(self.str_num):
            L = A[:, [self.str_indices[i] for _ in range(A.size(1))], :, :]
            R = A[:, [self.str_indices[(i + 1) % self.str_num] for
                      _ in range(A.size(1))], :, :]
            # A.index_fill_(1, LongTensor(list(blank_inds)), 1)
            AA = torch.cat((L, A, R), 3)
            AA.index_fill_(1, LongTensor(list(blank_inds)), 1)
            BB = torch.cat((L, B, R), 3)
            AAA[i, :, :, :, :] = AA
            BBB[i, :, :, :, :] = BB

        return {'A': AAA, 'A_paths': AB_paths, 'B': BBB,
                'B_paths': AB_paths, 'indices': remain_inds}


# generate randomly extended alphabet with skeletons along timestep
class ExHalfTskData(object):
    def __init__(self, str_indices, data_loader, fineSize,
                 max_dataset_size, rgb,
                 dict_test={}, blanks=0.7, skmode=1):
        self.skmode = skmode
        self.str_indices = str_indices
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.n_rgb = 3 if self.rgb else 1
        self.blanks = blanks
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
            w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
            w_offset: w_offset + self.fineSize]
        self.ab_size = int(A.size(1) / self.n_rgb)

        remain_inds = list(set(self.str_indices) - {-1})
        if remain_inds:
            self.str_num = len(self.str_indices)
            self.remainnum = len(remain_inds)
            self.blanknum = self.ab_size - self.remainnum
            blank_inds = list(
                set(range(self.ab_size)) - set(remain_inds))
        else:
            self.blanknum = int(self.blanks * A.size(1) / self.n_rgb)
            self.remainnum = int(
                A.size(1) / self.n_rgb) - self.blanknum
            self.str_num = self.remainnum
            # randomly remove some of the glyphs in input
            if not self.dict:
                blank_inds = np.random.permutation(
                    range(self.ab_size))[0:self.blanknum]
            else:
                file_name = list(map(lambda x: x.split("/")[-1],
                                  AB_paths))
                if len(file_name) > 1:
                    raise Exception('batch size should be 1')
                file_name = file_name[0]
                blank_inds = self.random_dict[file_name][
                            0:self.blanknum]
            remain_inds = list(set(range(self.ab_size))
                               - set(blank_inds))
            self.str_indices = remain_inds
            blank_inds = np.repeat(blank_inds, self.n_rgb)
        rgb_inds = np.tile(range(self.n_rgb), self.blanknum)
        blank_inds = blank_inds * self.n_rgb + rgb_inds

        AAA = A.unsqueeze(0)
        AAA = AAA.repeat([self.str_num, 1, 1, 1, 3])
        AAsk0 = AAA.clone()
        AAsk0.cpu()
        AAsk1 = AAsk0.clone()
        BBB = B.unsqueeze(0)
        BBB = BBB.repeat([self.str_num, 1, 1, 1, 3])
        BBsk0 = AAsk0.clone()
        BBsk1 = AAsk0.clone()
        for k in range(self.str_num):
            if not self.dict:
                L = A[:, [remain_inds[random.randint(
                    0, self.remainnum - 1)] for _ in
                          range(A.size(1))], :, :]
                R = A[:, [remain_inds[random.randint(
                    0, self.remainnum - 1)] for _ in
                          range(A.size(1))], :, :]
            else:
                L = A[:, [remain_inds[((self.random_dict[file_name][
                                           (i * 9 + 7) % A.size(1)]
                                       * 9 + 7) % self.remainnum)]
                          for i in range(A.size(1))], :, :]
                R = A[:, [remain_inds[((self.random_dict[file_name][
                                            (i * 9 - 7) % A.size(1)]
                                        * 9 + 7) % self.remainnum)]
                          for i in range(A.size(1))], :, :]
            # A.index_fill_(1, LongTensor(list(blank_inds)), 1)
            AA = torch.cat((L, A, R), 3)
            BB = AA.clone()
            AA.index_fill_(1, LongTensor(list(blank_inds)), 1)
            AAA[k, :, :, :, :] = AA
            BBB[k, :, :, :, :] = BB
            BB.cpu()

            if not self.skmode == 1:
                Bsk0 = BB.numpy()
                for i in range(A.size(0)):
                    for j in range(A.size(1)):
                        Bsk0[i, j, :, :] = skeletonize(Bsk0[i, j, :,
                                                       :] > 0)
                        Bsk0[i, j, :, :] = invert(Bsk0[i, j, :, :])
                Bsk0 = LongTensor(Bsk0)
                BBsk0[k, :, :, :, :] = Bsk0
                Ask0 = Bsk0.clone()
                Ask0.index_fill_(1, LongTensor(list(blank_inds)), 1)
                AAsk0[k, :, :, :, :] = Ask0

            if not self.skmode == 0:
                Bsk1 = BB.numpy()
                for i in range(A.size(0)):
                    for j in range(A.size(1)):
                        Bsk1[i, j, :, :] = invert(Bsk1[i, j, :, :])
                        Bsk1[i, j, :, :] = skeletonize(
                            Bsk1[i, j, :, :] > 0)
                        Bsk1[i, j, :, :] = invert(Bsk1[i, j, :, :])
                Bsk1 = LongTensor(Bsk1)
                BBsk1[k, :, :, :, :] = Bsk1
                Ask1 = Bsk1.clone()
                Ask1.index_fill_(1, LongTensor(list(blank_inds)), 1)
                AAsk1[k, :, :, :, :] = Ask1

        if self.skmode == 0:
            return {'A': AAA, 'A_paths': AB_paths, 'B': BBB,
                    'B_paths': AB_paths, 'Ask0': AAsk0, 'Bsk0':
                        BBsk0, 'indices': remain_inds}
        if self.skmode == 1:
            return {'A': AAA, 'A_paths': AB_paths, 'B': BBB,
                    'B_paths': AB_paths, 'Ask1': AAsk1, 'Bsk1':
                        BBsk1, 'indices': remain_inds}
        if self.skmode == 2:
            return {'A': AAA, 'A_paths': AB_paths, 'B': BBB,
                    'B_paths': AB_paths, 'Ask0': AAsk0, 'Bsk0': BBsk0,
                    'Ask1': AAsk1, 'Bsk1': BBsk1, 'indices':
                        remain_inds}


# generate alphabet for selected font
class PartialData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_base, fineSize, loadSize, max_dataset_size, phase, base_font=False, blanks=0):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_base = data_loader_base
        self.fineSize = fineSize
        self.loadSize = loadSize
        self.max_dataset_size = max_dataset_size
        self.blanks = blanks

        self.base_font = base_font
        if base_font:
            self.data_loader_base_iter = iter(self.data_loader_base)
            self.A_base,self.A_base_paths = next(self.data_loader_base_iter)
            print(self.A_base.size(3))
            self.A_base[0,:,:,:]=normalize_stack(self.A_base[0,:,:,:])
        else:
            self.A_base = []
        self.phase =phase
        if self.phase=='train':

            t_tensor =  transforms.Compose([
                transforms.ToTensor(),])
            t_topil = transforms.Compose([
                transforms.ToPILImage()])

            if self.base_font:
                for ind in range(self.A_base.size(1)):
                    A_base = t_topil(self.A_base[0,ind,:,:].unsqueeze(0))

    def __iter__(self):
        self.data_loader_iter_A = iter(self.data_loader_A)
        self.data_loader_iter_B = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration

        A, A_paths = next(self.data_loader_iter_A)
        B, B_paths = next(self.data_loader_iter_B)


        t_topil = transforms.Compose([
            transforms.ToPILImage()])
        t_scale = transforms.Compose([
            transforms.Resize(self.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        t_normal = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        t_tensor =  transforms.Compose([
            transforms.ToTensor(),])

        # observed_glyph = list(set(np.nonzero(1 - A[0, :, :, :])[:,
        #                           0]))
        # print(observed_glyph)
        for index in range(A.size(0)):
            A[index,:,:,:]=normalize_stack(A[index,:,:,:])
            B[index,:,:,:]=normalize_stack(B[index,:,:,:])
            BB=t_topil(B[index,:,:,:])
            # remove more of the glyphs to make prediction harder
            if self.blanks!=0:
                gt_glyph = [int(A_paths[index].split('/')[-1].split('.png')[0].split('_')[-1])]
                observed_glyph = list(set(np.nonzero(1-A[index,:,:,:])[:,0]) - set(gt_glyph))

                observed_glyph = np.random.permutation(observed_glyph)
                blank_nums = 1
                for i in range(blank_nums):
                    A[index,observed_glyph[i],:,:] = 1

        return {'A': A, 'A_paths': A_paths, 'B':B, 'B_paths':B_paths, 'A_base':self.A_base, 'indices': []}


# generate randomly extended alphabet for selected font
class EPartialData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_base, fineSize, loadSize, max_dataset_size, phase, base_font=False, blanks=0):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_base = data_loader_base
        self.fineSize = fineSize
        self.loadSize = loadSize
        self.max_dataset_size = max_dataset_size
        self.blanks = blanks

        self.base_font = base_font
        if base_font:
            self.data_loader_base_iter = iter(self.data_loader_base)
            self.A_base,self.A_base_paths = next(self.data_loader_base_iter)
            print(self.A_base.size(3))
            self.A_base[0,:,:,:]=normalize_stack(self.A_base[0,:,:,:])
        else:
            self.A_base = []
        self.phase =phase
        if self.phase=='train':

            t_tensor =  transforms.Compose([
                transforms.ToTensor(),])
            t_topil = transforms.Compose([
                transforms.ToPILImage()])

            if self.base_font:
                for ind in range(self.A_base.size(1)):
                    A_base = t_topil(self.A_base[0,ind,:,:].unsqueeze(0))

    def __iter__(self):
        self.data_loader_iter_A = iter(self.data_loader_A)
        self.data_loader_iter_B = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration

        A, A_paths = next(self.data_loader_iter_A)
        B, B_paths = next(self.data_loader_iter_B)

        t_topil = transforms.Compose([
            transforms.ToPILImage()])
        t_scale = transforms.Compose([
            transforms.Resize(self.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        t_normal = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        t_tensor = transforms.Compose([
            transforms.ToTensor(),])

        observed_glyph = list(set(np.nonzero(1 - A[0, :, :, :])[:,
                                   0]))
        observed_num = len(observed_glyph)
        # print(observed_glyph)
        for index in range(A.size(0)):
            A[index,:,:,:]=normalize_stack(A[index,:,:,:])
            B[index,:,:,:]=normalize_stack(B[index,:,:,:])
            BB=t_topil(B[index,:,:,:])
            # remove more of the glyphs to make prediction harder
            if self.blanks!=0:
                gt_glyph = [int(A_paths[index].split('/')[-1].split('.png')[0].split('_')[-1])]
                observed_glyph_ = list(set(np.nonzero(1 - A[index,
                                                          :, :, :])[:,0]) - set(gt_glyph))
                observed_glyph_ = np.random.permutation(
                    observed_glyph_)
                blank_nums = 1
                for i in range(blank_nums):
                    A[index,observed_glyph_[i],:,:] = 1
        # observed_glyph_ = np.random.permutation(observed_glyph)
        added_glyph = [
            observed_glyph[random.randint(0, observed_num-1)]
            for _ in range(A.size(1))]
        added_glyph_ = [added_glyph[i] for i in observed_glyph]
        L = A
        L[:, observed_glyph, :, :] = A[:, added_glyph_, :, :]
        # L_ = B
        # L_[:, :, :, :] = B[:, added_glyph, :, :]
        added_glyph = [
            observed_glyph[random.randint(0, observed_num - 1)]
            for _ in range(A.size(1))]
        added_glyph_ = [added_glyph[i] for i in observed_glyph]
        R = A
        R[:, observed_glyph, :, :] = A[:, added_glyph_, :, :]
        # R_ = B
        # R_[:, :, :, :] = B[:, added_glyph, :, :]
        AA = torch.cat((L, A, R), 3)
        # BB = torch.cat((L_, B, R_), 3)

        return {'A': AA, 'A_paths': A_paths, 'B': B,
                'B_paths': B_paths, 'A_base': self.A_base,
                'indices': observed_glyph}


# data loader for second stage with PartialData
class StackDataLoader(BaseDataLoader):
    """ a subset of the glyphs are observed and being used for transferring style
        train a pix2pix model conditioned on b/w glyphs
        to generate colored glyphs.
    """

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
                                 ])
        dic_phase = {'train':'Train', 'test':'Test'}
        # Dataset A
        dataset_A = ImageFolder(root=self.dataroot +'A/'+ opt.phase, transform=transform, return_paths=True, rgb=opt.rgb_in, fineSize=opt.fineSize, loadSize=opt.loadSize, font_trans=True, no_permutation=opt.no_permutation)
        len_A = len(dataset_A.imgs)
        shuffle_inds = np.random.permutation(len_A)

        dataset_B = ImageFolder(root=self.dataroot  + 'B/'+ opt.phase,
                              transform=transform, return_paths=True, rgb=opt.rgb_out,
                              fineSize=opt.fineSize, loadSize=opt.loadSize,
                              font_trans=False, no_permutation=opt.no_permutation)


        if len(dataset_A.imgs)!=len(dataset_B.imgs):
            raise Exception("number of images in source folder and target folder does not match")

        if (opt.partial and (not self.opt.serial_batches)):
            dataset_A.imgs = [dataset_A.imgs[i] for i in shuffle_inds]
            dataset_B.imgs = [dataset_B.imgs[i] for i in shuffle_inds]
            dataset_A.img_crop = [dataset_A.img_crop[i] for i in shuffle_inds]
            dataset_B.img_crop = [dataset_B.img_crop[i] for i in shuffle_inds]
            shuffle = False
        else:
            shuffle = not self.opt.serial_batches
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads), drop_last=True)


        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads), drop_last=True)

        if opt.base_font:
            # Read and apply transformation on the BASE font

            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        else:
            data_loader_base=None

        self.dataset_A = dataset_A
        self._data = PartialData(data_loader_A, data_loader_B,
                                 data_loader_base, opt.fineSize,
                                 opt.loadSize, opt.max_dataset_size,
                                 opt.phase, opt.base_font, opt.blanks)
    def name(self):
        return 'StackDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset_A), self.opt.max_dataset_size)


# data loader for second stage with EPartialData
class EStackDataLoader(BaseDataLoader):
    """ a subset of the glyphs are observed and being used for transferring style
        train a pix2pix model conditioned on b/w glyphs
        to generate colored glyphs.
    """

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
                                 ])
        dic_phase = {'train':'Train', 'test':'Test'}
        # Dataset A
        dataset_A = ImageFolder(root=self.dataroot +'A/'+ opt.phase, transform=transform, return_paths=True, rgb=opt.rgb_in, fineSize=opt.fineSize, loadSize=opt.loadSize, font_trans=True, no_permutation=opt.no_permutation)
        len_A = len(dataset_A.imgs)
        shuffle_inds = np.random.permutation(len_A)

        dataset_B = ImageFolder(root=self.dataroot  + 'B/'+ opt.phase,
                              transform=transform, return_paths=True, rgb=opt.rgb_out,
                              fineSize=opt.fineSize, loadSize=opt.loadSize,
                              font_trans=False, no_permutation=opt.no_permutation)


        if len(dataset_A.imgs)!=len(dataset_B.imgs):
            raise Exception("number of images in source folder and target folder does not match")

        if (opt.partial and (not self.opt.serial_batches)):
            dataset_A.imgs = [dataset_A.imgs[i] for i in shuffle_inds]
            dataset_B.imgs = [dataset_B.imgs[i] for i in shuffle_inds]
            dataset_A.img_crop = [dataset_A.img_crop[i] for i in shuffle_inds]
            dataset_B.img_crop = [dataset_B.img_crop[i] for i in shuffle_inds]
            shuffle = False
        else:
            shuffle = not self.opt.serial_batches
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads), drop_last=True)


        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads), drop_last=True)

        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        else:
            data_loader_base=None

        self.dataset_A = dataset_A
        self._data = EPartialData(data_loader_A, data_loader_B,
                                 data_loader_base, opt.fineSize,
                                 opt.loadSize, opt.max_dataset_size,
                                 opt.phase, opt.base_font, opt.blanks)
    def name(self):
        return 'StackDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset_A), self.opt.max_dataset_size)


class PartialDataLoader(BaseDataLoader):
    """ a subset of the glyphs are observed and being used for training stlye
        train a pix2pix model conditioned on b/w glyphs
        to generate colored glyphs.
        In the pix2pix model it is simmilar to the unaligned data class.
    """

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dic_phase = {'train':'Train', 'test':'Test'}
        # Dataset A

        dataset_A = ImageFolder(root=self.dataroot + 'A/' + opt.phase,
                                transform=transform,
                                return_paths=True, rgb=opt.rgb,
                                fineSize=opt.fineSize,
                                loadSize=opt.loadSize,
                                font_trans=False,
                                no_permutation=opt.no_permutation)
        len_A = len(dataset_A.imgs)
        if not opt.no_permutation:
            shuffle_inds = np.random.permutation(len_A)
        else:
            shuffle_inds = range(len_A)

        dataset_B = ImageFolder(root=self.dataroot + 'B/' + opt.phase,
                                transform=transform,
                                return_paths=True,
                                rgb=opt.rgb, fineSize=opt.fineSize,
                                loadSize=opt.loadSize,
                                font_trans=False,
                                no_permutation=opt.no_permutation)

        if len(dataset_A.imgs)!=len(dataset_B.imgs):
            raise Exception("number of images in source folder "
                            "and target folder does not match")

        if (opt.partial and (not self.opt.serial_batches)):
            dataset_A.imgs = [dataset_A.imgs[i] for i in shuffle_inds]
            dataset_B.imgs = [dataset_B.imgs[i] for i in shuffle_inds]
            dataset_A.img_crop = [dataset_A.img_crop[i] for
                                  i in shuffle_inds]
            dataset_B.img_crop = [dataset_B.img_crop[i] for
                                  i in shuffle_inds]
            shuffle = False
        else:
            shuffle = not self.opt.serial_batches
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads), drop_last=True)


        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads), drop_last=True)

        if opt.base_font:
            #Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=opt.base_root,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base, batch_size=1, shuffle=False,
                num_workers=int(self.opt.nThreads))
        else:
            data_loader_base = None

        self.dataset_A = dataset_A
        self._data = PartialData(data_loader_A,data_loader_B,
                                 data_loader_base, opt.fineSize,
                                 opt.loadSize, opt.max_dataset_size,
                                 opt.phase, opt.base_font)

    def name(self):
        return 'PartialDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset_A), self.opt.max_dataset_size)


# data loaders
class DataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=os.path.join(self.dataroot, opt.phase),
                              transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                              fineSize=opt.fineSize, loadSize=opt.loadSize)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads), drop_last=True)

        self.dataset = dataset
        dict_inds = {}
        test_dict = self.dataroot+'/test_dict/dict.pkl'
        # if opt.phase == 'train':
        if opt.phase.startswith('test'):
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb'))
            else:
                warnings.warn('Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')

        str_indices = self.opt.str_input_indices
        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.base_font, opt.blanks)
        else:
            self._data = Data(str_indices, data_loader,
                              opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.blanks)

    def name(self):
        return 'DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class skDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=os.path.join(self.dataroot, opt.phase),
                              transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                              fineSize=opt.fineSize, loadSize=opt.loadSize)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads), drop_last=True)

        self.dataset = dataset
        dict_inds = {}
        test_dict = self.dataroot+'/test_dict/dict.pkl'
        # if opt.phase == 'train':
        if opt.phase.startswith('test'):
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb', ))
            else:
                warnings.warn('Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')

        str_indices = self.opt.str_input_indices
        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.base_font, opt.blanks)
        else:
            self._data = skData(str_indices, data_loader,
                                opt.fineSize, opt.max_dataset_size,
                                opt.rgb, dict_inds, opt.blanks,
                                skmode=opt.skmode)

    def name(self):
        return 'skDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class EHskDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=os.path.join(self.dataroot, opt.phase),
                              transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                              fineSize=opt.fineSize, loadSize=opt.loadSize)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads), drop_last=True)

        self.dataset = dataset
        dict_inds = {}
        test_dict = self.dataroot+'/test_dict/dict.pkl'
        # if opt.phase == 'train':
        if opt.phase.startswith('test'):
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb', ))
            else:
                warnings.warn('Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')

        str_indices = self.opt.str_input_indices
        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.base_font, opt.blanks)
        else:
            self._data = EHskData(str_indices, data_loader,
                                opt.fineSize, opt.max_dataset_size,
                                opt.rgb, dict_inds, opt.blanks,
                                skmode=opt.skmode)

    def name(self):
        return 'EHskDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class ExHalfDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=os.path.join(self.dataroot, opt.phase),
                              transform=transform, return_paths=True,
                              font_trans=(not opt.flat), rgb=opt.rgb,
                              fineSize=opt.fineSize, loadSize=opt.loadSize)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batchSize,
            shuffle=(not self.opt.serial_batches),
            num_workers=int(self.opt.nThreads), drop_last=True)

        self.dataset = dataset
        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        dict_inds = {}
        test_dict = self.dataroot + '/test_dict/dict.pkl'
        if opt.phase == 'test':
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb', ))
            else:
                warnings.warn(
                    'Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')
        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize,
                                  opt.max_dataset_size, opt.rgb, dict_inds,
                                  opt.base_font, opt.blanks)
        else:
            self._data = ExHalfData(data_loader, opt.fineSize,
                                 opt.max_dataset_size,
                              opt.rgb, dict_inds, opt.blanks)

    def name(self):
        return 'ExHalfDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class ExHalfTDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=os.path.join(self.dataroot, opt.phase),
                              transform=transform, return_paths=True,
                              font_trans=(not opt.flat), rgb=opt.rgb,
                              fineSize=opt.fineSize, loadSize=opt.loadSize)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batchSize,
            shuffle=(not self.opt.serial_batches),
            num_workers=int(self.opt.nThreads), drop_last=True)

        self.dataset = dataset
        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        dict_inds = {}
        test_dict = self.dataroot + '/test_dict/dict.pkl'
        if opt.phase == 'test':
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb',))
            else:
                warnings.warn(
                    'Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')

        str_indices = self.opt.str_input_indices
        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize,
                                  opt.max_dataset_size, opt.rgb, dict_inds,
                                  opt.base_font, opt.blanks)
        else:
            self._data = ExHalfTData(str_indices, data_loader,
                                     opt.fineSize,
                                      opt.max_dataset_size, opt.rgb,
                                      dict_inds, opt.blanks)

    def name(self):
        return 'ExHalf_tDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class ExHalfTskDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=os.path.join(self.dataroot, opt.phase),
                              transform=transform, return_paths=True,
                              font_trans=(not opt.flat), rgb=opt.rgb,
                              fineSize=opt.fineSize, loadSize=opt.loadSize)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batchSize,
            shuffle=(not self.opt.serial_batches),
            num_workers=int(self.opt.nThreads), drop_last=True)

        self.dataset = dataset
        dict_inds = {}
        test_dict = self.dataroot + '/test_dict/dict.pkl'
        if opt.phase == 'test':
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb',))
            else:
                warnings.warn(
                    'Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')

        str_indices = self.opt.str_input_indices
        if opt.base_font:
            # Read and apply transformation on the BASE font
            dataset_base = ImageFolder(root=self.baseroot,
                                       transform=transform,
                                       return_paths=True,
                                       font_trans=True, rgb=opt.rgb,
                                       fineSize=opt.fineSize,
                                       loadSize=opt.loadSize)
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize,
                                  opt.max_dataset_size, opt.rgb, dict_inds,
                                  opt.base_font, opt.blanks)
        else:
            self._data = ExHalfTskData(str_indices, data_loader,
                                       opt.fineSize,
                                       opt.max_dataset_size, opt.rgb,
                                       dict_inds, opt.blanks,
                                       skmode=opt.skmode)

    def name(self):
        return 'ExHalfTskDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)



