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
this_path = os.path.dirname(os.path.realpath(__file__))
project_root = this_path.rpartition("xifontgan")[0]
sys.path.insert(0, project_root)
from xifontgan.util.indexing import str2index


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
        self.parser.add_argument('--blanks', nargs='+', type=float,
                                 default=[0.7], help='blank rate')
        self.parser.add_argument('--grps', type=int, default=26,
                                 help='blank rate')
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


def folder2model(folder):
    if folder.startswith('cGAN'):
        return 'cGAN'
    elif folder.startswith('EcGAN'):
        return 'EcGAN'
    elif folder.startswith('sk1GAN'):
        return 'sk1GAN'
    elif folder.startswith('EskGAN'):
        return 'EskGAN'
    else:
        return folder


def folder2tag(folder):
    model = folder2model(folder)
    tags = {"cGAN": ["real_A", "real_B", "fake_B"], "cycleGAN": [
        "real_A", "refake_A", "real_B"], "EcGAN": [
        "real_A", "fake_B", "real_B"], "LSTMcGAN": [
        "real_A", "fake_B", "real_B"]}
    return tags.get(model, [])


def folder2gt():
    return ["real_A", "real_B"]
    # return ["real_B"]


def folder2gt_sk1():
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

def folder2result(folder):
    model = folder2model(folder)
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


def folder2sk(folder):
    model = folder2model(folder)
    tags = {"gt": [],
            "cGAN": [],
            "cycleGAN": [],
            "EcGAN": [],
            "LSTMcGAN": [],
            "sk1GAN": [],
            "zi2zi": [],
            "EskGAN": ["fake_Bsk1"]}
    return tags.get(model, [])


def folder2skgt(folder):
    model = folder2model(folder)
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
    ch = opt.grps

    for font in opt.font:
        models = ""
        file_dirs = []
        files = []
        zip_files = []
        zip_file_listings = []
        imgfiles = []
        imgs = []
        img_sks = []
        hint_ = 0

        gt_list = []
        for blank in opt.blanks:
            gt_dir = os.path.join(project_root, "xifontgan",
                                  "results", opt.dataset,
                                  'blanks_%s' % blank)
            gt_file = os.path.join(gt_dir, "gt.zip")
            Egt_file = os.path.join(gt_dir, "Egt.zip")
            gt_sk_file = os.path.join(gt_dir, "gt_sk.zip")
            Egt_sk_file = os.path.join(gt_dir, "Egt_sk.zip")

            gt_zipfile = ZipFile(gt_file, "r")
            gt_list += [[Image.open(gt_zipfile.open(font + '_' + j
                                                    + '.png')) for
                         j in folder2gt()]]
        # imgs += gt_list
        imgs += [gt_list[0][1]]

        k = 0
        for i in opt.model:
            j = i.partition("_")
            if not models:
                models = j[0]
            else:
                models = models + "&" + j[0]
            file_dir = os.path.join(project_root, "xifontgan",
                                    "results", opt.dataset + "_" + i)
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
                zip_file_listing = 'NA'
                # with open(file_zip_pickle, 'rb') as f:
                #     zip_file_listing = pickle.load(f,
                #                                    encoding='utf-8')
            zip_file_listings += [zip_file_listing]
            # if hint_ == 0:
            #     imgs += [Image.open(zip_file.open('images/' + font +
            #              '_' + j + '.png')) for j in folder2gt()]
            #     hint_ = 1
            img_ = [Image.open(zip_file.open('images/' + font +
                     '_' + j + '.png')) for j in folder2result(i)]

            imgs += [redbox_gt(gt_list[k][0], j, ch) for j in img_]
            k += 1

        outputfile = os.path.join(outputdir,
                                  models + "_" + font + "_blanks.png")
        # tags = folder2tag(opt.model)
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

        pad = 1
        max_width = max(widths) + 2 * ch * pad
        width_ = int(max_width / ch)
        total_height = sum([(x+2*pad) for x in list(heights)])
        height_ = 0
        new_img = Image.new('RGB', (max_width, total_height),
                            color=(255,255,255))
        x_offset = 0
        y_offset = 0
        for img in imgs:
            width, height = img.size
            width = int(width / ch)
            for grp in range(ch):
                img_ = img.crop((grp*width, 0, (grp+1)*width, height))
                new_img.paste(img_, (grp*width_+pad, height_+pad))
            height_ += height + 2 * pad
            # new_img.paste(im, (x_offset, y_offset))
            # x_offset += im.size[0]
            # y_offset += im.size[1]
        new_img.save(outputfile)

        if opt.string:
            str_dir = os.path.join(outputdir, "string000")
            # os.makedirs(str_dir)
            str_txt = os.path.join(str_dir, "string.txt")
            str_file = os.path.join(str_dir, models + "_" + font +
                                    ".png")
            str_indices = str2index(opt.string, "ENfull")
            str_len = len(str_indices)
            str_img = Image.new('RGB', (str_len * width_,
                                        total_height),
                                color=(255, 255, 255))
            str_num = 0
            print(str_indices)
            for index in str_indices:
                if index > -1:
                    img_ = new_img.crop((index*width_, 0,
                                        (index+1)*width_,
                                        total_height))
                    str_img.paste(img_, (str_num * width_, 0))
                str_num += 1
            str_img.save(str_file)


if __name__ == "__main__":
    main()


