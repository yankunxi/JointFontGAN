import os
import sys
import shutil
import time
import configparser
from zipfile import ZipFile
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
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
        self.parser.add_argument('--blanks', type=float, default=0.7,
                                 help='blank rate')
        self.parser.add_argument('--grps', type=int, default=26,
                                 help='blank rate')

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


def model2gt():
    return ["real_A", "real_B"]
    # return ["real_B"]


def model2gt_sk1():
    return ["real_Ask1", "real_Bsk1"]
    # return ["real_Bsk1"]


def redbox_gt(a, b, ch):
    pad = 2
    w, h = a.size

    pix_a = np.asarray(a)
    new_img = Image.new('RGB', (w+2*ch*pad, h+2*pad),
                        color=(255, 255, 255))
    w = int(w/ch)
    b = b.convert("RGBA")
    img_draw = ImageDraw.Draw(new_img)
    for i in range(ch):
        im_ = b.crop((i * w, 0, (i + 1) * w, h))
        new_img.paste(im_, (i*(w+2*pad)+pad, pad))
    for i in range(ch):
        if sum(sum(255 - pix_a[:, i*h:(i+1)*h-1])):
            img_draw.rectangle([i*(w+2*pad), 0, (i+1)*(w+2*pad)-1,
                                h+2*pad-1], outline=(175, 0, 0, 255),
                               width=pad)
    return new_img

def model2result(model):
    tags = {"gt": [],
            "zi2zi": ["fake_B"],
            "cGAN": ["fake_B"],
            "cycleGAN": ["refake_A"],
            "EcGAN": ["fake_B"],
            "LSTMcGAN": ["fake_B"],
            "sk1GAN3": ["fake_B"],
            "sk1GAN": ["fake_B"],
            "EskGAN": ["fake_B"]}
    return tags.get(model, [])


def model2sk(model):
    tags = {"gt": [],
            "cGAN": [],
            "cycleGAN": [],
            "EcGAN": [],
            "LSTMcGAN": [],
            "sk1GAN3": ["fake_Bsk1"],
            "sk1GAN2": [],
            "zi2zi": [],
            "EskGAN": ["fake_Bsk1"]}
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
                             "_show", opt.dataset + "_gt")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    ch = opt.grps

    file_dirs = []
    files = []
    zip_files = []
    zip_file_listings = []
    imgfiles = []
    imgs = []
    img_sks = []
    hint_ = 0

    gt_dir = os.path.join(project_root, "xifontgan", "results",
                          opt.dataset, 'blanks_%s' % opt.blanks)
    gt_file = os.path.join(gt_dir, "gt.zip")
    Egt_file = os.path.join(gt_dir, "Egt.zip")
    gt_sk_file = os.path.join(gt_dir, "gt_sk.zip")
    Egt_sk_file = os.path.join(gt_dir, "Egt_sk.zip")
    print(gt_file)
    for font in opt.font:
        models = ""

        gt_zipfile = ZipFile(gt_file, "r")
        gt_list = [Image.open(gt_zipfile.open(font + '_' + j +
                                              '.png')) for j in
                   model2gt()]
        # imgs += gt_list
        # imgs += [redbox_gt(gt_list[0], gt_list[1], ch)]
        imgs += [gt_list[1]]

    outputfile = os.path.join(outputdir, "dataset.png")

    widths, heights = zip(*(i.size for i in imgs))

    pad = 1
    max_width = max(widths) + 2 * ch * pad
    width_ = int(max_width / ch)
    total_height = sum([(x + 2 * pad) for x in list(heights)])
    height_ = 0
    new_img = Image.new('RGB', (max_width, total_height),
                        color=(255, 255, 255))
    x_offset = 0
    y_offset = 0
    for im in imgs:
        width, height = im.size
        width = int(width / ch)
        for grp in range(ch):
            im_ = im.crop((grp * width, 0, (grp + 1) * width, height))
            new_img.paste(im_, (grp * width_ + pad, height_ + pad))
        height_ += height + 2 * pad
        # new_img.paste(im, (x_offset, y_offset))
        # x_offset += im.size[0]
        # y_offset += im.size[1]
    new_img.save(outputfile)




if __name__ == "__main__":
    main()


