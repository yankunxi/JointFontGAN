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
        # self.parser.add_argument('--font', nargs='+', required=True,
        #                          help='font to show')
        self.parser.add_argument('--font', required=True,
                                 help='font to show')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


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
        temp = 255 - pix_a[:, i*h:(i+1)*h-1]
        if temp.any():
            img_draw.rectangle([i*(w+2*pad), 0, (i+1)*(w+2*pad)-1,
                                h+2*pad-1], outline=(175, 0, 0, 255),
                               width=pad)
    return new_img


def main():
    opt = ShowOptions().parse()
    use_auxiliary = 1
    if use_auxiliary == 1:
        result_root = os.path.join(
            "/mnt/Files/XIauxiliary/XIcodes", project_root.partition(
                "XIcodes/")[2],
            "xifontgan/results/public_web_fonts")
    else:
        result_root = os.path.join(project_root,
                                   "xifontgan/results/public_web_fonts")
    result_cGANbf_dir = os.path.join(result_root, opt.font,
                                     "StackGAN_cGANbf/test_400+300@300/images",
                                     opt.font+"_fake_B.png")
    result_EcGAN_dir = os.path.join(result_root, opt.font,
                                    "StackGAN_EcGAN/test_400+300@300/images",
                                    opt.font + "_fake_B.png")
    img_cGANbf = Image.open(result_cGANbf_dir)
    img_EcGAN = Image.open(result_EcGAN_dir)

    data_root = "/mnt/Files/XIremote/OneDrive - Wayne State University/XIdataset/font/English/public_web_fonts/"
    data_A_dir = os.path.join(data_root, opt.font,  "A/test",
                              opt.font+".png")
    data_B_dir = os.path.join(data_root, opt.font, "B/test",
                              opt.font + ".png")
    img_A = Image.open(data_A_dir)
    img_B = Image.open(data_B_dir)

    img_blank = Image.new('RGB', (26*64, 64),
                            color=(255, 255, 255))

    new_img = Image.new('RGB', (26*68, 3*68),
                        color=(255, 255, 255))
    a = redbox_gt(img_A, img_B, 26)
    new_img.paste(a, (0, 0))
    b = redbox_gt(img_blank, img_cGANbf, 26)
    new_img.paste(b, (0, 68))
    c = redbox_gt(img_blank, img_EcGAN, 26)
    new_img.paste(c, (0, 136))

    to_file = os.path.join(result_root, "files", opt.font+".png")
    new_img.save(to_file)


if __name__ == "__main__":
    main()
