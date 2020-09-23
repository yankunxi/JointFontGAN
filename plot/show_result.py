import os
import sys
import shutil
import time
import configparser
from zipfile import ZipFile
import numpy as np
from PIL import Image, ImageEnhance
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
                                 default=[500],
                                 help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', nargs='+', type=int,
                                 default=[100],
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--which_epoch', nargs='+', type=int,
                                 default=[600], help='# of iter ' \
                                                    'wanted')
        self.parser.add_argument('--align', type=str, default='row',
                                 help='how do you want it to be aligned')
        self.parser.add_argument('--string', type=str, default='',
                                 help='some strings to show')
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


def model2tag(model):
    tags = {"cGAN": ["real_A", "real_B", "fake_B"], "cycleGAN": [
        "real_A", "refake_A", "real_B"], "EcGAN": [
        "real_A", "fake_B", "real_B"], "LSTMcGAN": [
        "real_A", "fake_B", "real_B"]}
    return tags.get(model, [])

def model2gt(model):
    return ["real_A", "real_B"]

def model2result(model):
    tags = {"zi2zi": ["fake_B"],
            "cGAN": ["fake_B"],
            "cycleGAN": ["refake_A"],
            "EcGAN": ["fake_B"],
            "LSTMcGAN": ["fake_B"],
            "sk1GAN3": ["fake_B"],
            "sk1GAN2": ["fake_B"],
            "EskGAN3": ["fake_B"]}
    return tags.get(model, [])

def model2sk(model):
    tags = {"cGAN": [],
            "cycleGAN": [],
            "EcGAN": [],
            "LSTMcGAN": [],
            "sk1GAN3": ["fake_Bsk1"],
            "sk1GAN2": [],
            "zi2zi": [],
            "EskGAN3": ["fake_Bsk1"]}
    return tags.get(model, [])

def model2skgt(model):
    return ["real_Ask1", "real_Bsk1"]

def main():
    opt = ShowOptions().parse()
    if opt.which_epoch[0] > opt.niter[0]:
        epoch_str = str(opt.niter[0]) + "+" + \
                    str(opt.which_epoch[0] - opt.niter[0]) + '@' + \
                    str(opt.niter_decay[0])
    print(opt.font)
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]

    outputdir = os.path.join(project_root, "xifontgan", "results",
                             "_show", opt.dataset + "_" + opt.phase
                             + "_" + epoch_str)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    gt_show = model2gt(opt.model[0])
    for font in opt.font:
        models = ""
        file_dirs = []
        files = []
        zip_files = []
        zip_file_listings = []
        imgfiles = []
        imgs = []
        hint_ = 0
        for i in opt.model:
            models = models + " " + i
            file_dir = os.path.join(project_root, "xifontgan", "results",
                               opt.dataset + "_" + i)
            file = os.path.join(file_dir, opt.phase + "_" +
                                epoch_str + ".zip")
            zip_file = ZipFile(file, "r")
            zip_files += [zip_file]
            file_zip_pickle = os.path.join(file_dir, opt.phase + "_" +
                                           epoch_str + "_ziplist.pkl")
            print(file)
            if not os.path.isfile(file_zip_pickle):
                with open(file_zip_pickle, 'wb') as f:
                    zip_file_listing = zip_file.namelist()
                    pickle.dump(zip_file_listing, f)
            else:
                with open(file_zip_pickle, 'rb') as f:
                    zip_file_listing = pickle.load(f,
                                                   encoding='utf-8')
            zip_file_listings += [zip_file_listing]
            if hint_ == 0:
                imgs += [Image.open(zip_file.open('images/' + font +
                         '_' + j + '.png')) for j in model2gt(i)]
                hint_ = 1
            imgs += [Image.open(zip_file.open('images/' + font +
                     '_' + j + '.png')) for j in model2result(i)]
            for j in model2sk(i):
                temp_img = Image.open(zip_file.open('images/' + font +
                                                    '_' + j + '.png'))
                temp_img = ImageEnhance.Contrast(temp_img).enhance(4)
                temp_img = ImageEnhance.Brightness(
                    temp_img).enhance(1.1)
                temp_img = ImageEnhance.Sharpness(temp_img).enhance(2)
                imgs += [temp_img]

        outputfile = os.path.join(outputdir,
                                  models + "_" + font + ".png")
        # tags = model2tag(opt.model)
        # gt_list = ['real_A', 'real_B']

        # for i in op_list:
        #     imgfiles = imgfiles + [
        #         file_zip.open('images/' + font + '_' + i
        #                       + '.png')]
        # imgfile1 = file_zip.open('images/1000hurt.0.0_fake_B.png')
        # imgfile2 = file_zip.open('images/1000hurt.0.0_real_B.png')
        # imgfile3 = file_zip.open('images/1000hurt.0.0_real_EA.png')
        # imgfiles = [imgfile1, imgfile2, imgfile3]

        widths, heights = zip(*(i.size for i in imgs))
        total_width = max(widths)
        max_height = sum(heights)
        new_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        y_offset = 0
        for im in imgs:
            new_img.paste(im, (x_offset, y_offset))
            # x_offset += im.size[0]
            y_offset += im.size[1]
        new_img.save(outputfile)



if __name__ == "__main__":
    main()


