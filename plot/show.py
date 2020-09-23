import os
import sys
import shutil
import time
import configparser
from zipfile import ZipFile
import numpy as np
from PIL import Image
import pickle
import argparse


class ShowOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        # get computer information
        self.computer_name = os.uname()[1]

    def initialize(self):
        self.parser.add_argument('--dataset', default="Capitals64",
                                 help='dataset to images')
        self.parser.add_argument('--model', nargs='+', required=True,
                                 help='model to show')
        self.parser.add_argument('--phase', type=str, default='test',
                                 help='train, val, test, etc')
        self.parser.add_argument('--font', nargs='+', required=True,
                                 help='font to show')
        self.parser.add_argument('--niter', nargs='+', type=int,
                                 default=500,
                                 help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', nargs='+', type=int,
                                 default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--which_epoch', nargs='+', type=int,
                                 default=600, help='# of iter wanted')
        self.parser.add_argument('--align', type=str, default='col',
                                 help='how do you want it to be aligned')
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


def model2tag(model):
    tags = {"cGAN": ["real_A", "fake_B", "real_B"], "cycleGAN": [
        "real_A", "refake_A", "real_B"], "EcGAN": [
        "real_A", "fake_B", "real_B"], "LSTMcGAN": [
        "real_A", "fake_B", "real_B"]}
    return tags.get(model, [])


def main():
    opt = ShowOptions().parse()
    if opt.which_epoch[0] > opt.niter[0]:
        epoch_str = str(opt.niter[0]) + "+" + \
                    str(opt.which_epoch[0] - opt.niter[0]) + '@' + \
                    str(opt.niter_decay[0])
    print(opt.font)
    for font in opt.font:
        this_path = os.path.dirname(os.path.realpath(__file__))
        project_root = this_path.rpartition("xifontgan")[0]
        filedir = os.path.join(project_root, "xifontgan", "results",
                               opt.dataset + "_" + opt.model[0])
        file = os.path.join(filedir,
                            opt.phase + "_" + epoch_str + ".zip")
        outputdir = os.path.join(project_root, "xifontgan", "results",
                                 "_show", opt.dataset + "_" +
                                 opt.model[0])
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        outputfile = os.path.join(outputdir,
                                  opt.phase + "_" + epoch_str +
                                  "_" + font + ".png")

        file_zip = ZipFile(file, "r")

        file_zip_pickle = os.path.join(filedir, opt.phase + "_" +
                                       epoch_str + "_ziplist.pkl")
        print(file)
        if not os.path.isfile(file_zip_pickle):
            with open(file_zip_pickle, 'wb') as f:
                file_zip_list = file_zip.namelist()
                pickle.dump(file_zip_list, f)
        else:
            with open(file_zip_pickle, 'rb') as f:
                file_zip_list = pickle.load(f, encoding='utf-8')

        # tags = model2tag(opt.model)
        # gt_list = ['real_A', 'real_B']
        gt_list = model2tag(opt.model[0])
        op_list= []
        # op_list = ['fake_B']

        imgfiles = [file_zip.open('images/' + font + '_' + i +
                                  '.png') for i in gt_list]
        for i in op_list:
            imgfiles = imgfiles + [
                file_zip.open('images/' + font + '_' + i
                              + '.png')]
        # imgfile1 = file_zip.open('images/1000hurt.0.0_fake_B.png')
        # imgfile2 = file_zip.open('images/1000hurt.0.0_real_B.png')
        # imgfile3 = file_zip.open('images/1000hurt.0.0_real_A.png')
        # imgfiles = [imgfile1, imgfile2, imgfile3]
        imgs = [Image.open(i) for i in imgfiles]
        widths, heights = zip(*(i.size for i in imgs))

        total_width = sum(widths)
        max_height = max(heights)

        new_img = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in imgs:
            new_img.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_img.save(outputfile)



if __name__ == "__main__":
    main()


