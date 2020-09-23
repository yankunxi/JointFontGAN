################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################

import argparse
import os
from xifontgan.util import util
import configparser


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        # get computer information
        self.computer_name = os.uname()[1]
        # get project root
        this_path = os.path.dirname(os.path.realpath(__file__))
        self.project_root = os.path.dirname(this_path)
        # read from files in config folder
        self.config_parser = configparser.SafeConfigParser()
        self.config_file_path = os.path.join(".", "config", "device")
        self.writing_file = False

    def read_or_set(self, section, option, value):
        if not self.config_parser.has_option(section, option):
            self.writing_file = True
            self.config_parser.set(section, option, value)
            print("!!! writing section [" + section + "], option [" + option + "], value [" + value + "]")
        else:
            print("reading section [" + section + "], option [" + option + "], value [" +
                  self.config_parser.get(section, option) + "]")

    def config_from_file(self):
        # read configuration from files
        if not os.path.exists(self.config_file_path):
            print("!!! creating configure file: [" + self.config_file_path + "]")
            self.writing_file = True
        else:
            self.config_parser.read(self.config_file_path)
        if not self.config_parser.has_section(self.computer_name):
            self.config_parser.add_section(self.computer_name)
        self.read_or_set(self.computer_name, "everything_root", self.project_root)
        self.read_or_set(self.computer_name, "use_auxiliary", "False")
        self.read_or_set(self.computer_name, "auxiliary_root", self.project_root)
        self.read_or_set(self.computer_name, "use_buffer", "False")
        self.read_or_set(self.computer_name, "buffer_root", self.project_root)
        if self.writing_file:
            with open(self.config_file_path, "wb") as config_file:
                self.config_parser.write(config_file)
                config_file.close()

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have sub-folders trainA, '
                                                                   'trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=26, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=26, help='# of output image channels')
        self.parser.add_argument('--grps', type=int, default=26, help='# of input groups')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--nif', type=int, default=32, help='# of transformation filters on top of input and '
                                                                     'output')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks',
                                 help='selects model to use for netG')
        self.parser.add_argument('--which_model_preNet', type=str, default='none',
                                 help='none/2_layers? selects model for prenetwork on top of input and prediction')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--align_data', action='store_true',
                                help='if True, the datasets are loaded from "test" and "train" directories and the '
                                     'data pairs are aligned')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, one_direction_test, pix2pix, ...')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--use_dropout1', action='store_true', help='use dropout for the generator in OrnaNet')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory '
                                      'contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--conditional', action='store_true', help='feed input to the discriminator')
        self.parser.add_argument('--conv3d', action='store_true', help='separate channels by 3d convolution?')
        self.parser.add_argument('--blanks', type=float, default=0.7,
                                 help='max ratio (in 26) of the number of glyphs to be blank')
        self.parser.add_argument('--rgb', action='store_true', help='consider all three RGB channels')
        self.parser.add_argument('--rgb_in', action='store_true', help='consider all three RGB channels for input')
        self.parser.add_argument('--rgb_out', action='store_true', help='consider all three RGB channels for output')
        self.parser.add_argument('--partial', action='store_true',
                                 help='have access to the ground truth of a subset of glyphs')
        self.parser.add_argument('--input_nc_1', type=int, default=3,
                                 help='# of input image channels in the 2nd network')
        self.parser.add_argument('--output_nc_1', type=int, default=3,
                                 help='# of output image channels in the 2nd network')
        self.parser.add_argument('--stack', action='store_true', help='have stacked networks?')
        self.parser.add_argument('--no_Style2Glyph', action='store_true',
                                 help='do not want to back prop from the StlyeNet to the GlyphNet')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--no_permutation', action='store_true',
                                 help='do not have random images in each batch')
        self.parser.add_argument('--base_font', action='store_true',
                                 help='use a base font for using in the conditional GAN')
        self.parser.add_argument('--base_root', default='',
                                 help='path to a base font : a simple grayscale font image containing all 26 glyphs')
        self.parser.add_argument('--print_weights', action='store_true', help='print initial weights of the netG1 network')
        self.parser.add_argument('--orna', action='store_true', help='only consider OrnaNet and should have full b/w '
                                                                     'inputs in A')
        self.parser.add_argument('--flat', action='store_true', help='consider input image as a flat image')

        self.initialized = True
        # set arguments from files
        self.config_from_file()
        self.parser.add_argument('--everything_root', type=str,
                                 default=self.config_parser.get(self.computer_name, "everything_root"),
                                 help='the coding root for project backup')
        self.parser.add_argument('--use_auxiliary', type=bool,
                                 default=self.config_parser.getboolean(self.computer_name, "use_auxiliary"),
                                 help='do we use auxiliary?')
        self.parser.add_argument('--auxiliary_root', type=str,
                                 default=self.config_parser.get(self.computer_name, "auxiliary_root"),
                                 help='the auxiliary root for project auxiliary')
        self.parser.add_argument('--use_buffer', type=bool,
                                 default=self.config_parser.getboolean(self.computer_name, "use_buffer"),
                                 help='do we use buffer?')
        self.parser.add_argument('--buffer_root', type=str,
                                 default=self.config_parser.get(self.computer_name, "buffer_root"),
                                 help='the buffer root for project buffer')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.project_root = self.project_root
        self.opt.computer_name = self.computer_name


        if self.opt.gpu_ids.find(' ')==-1:
            str_ids = self.opt.gpu_ids.split(',')
            self.opt.gpu_ids = []
            if len(str_ids)>=1:
                for str_id in str_ids:
                    id = int(str_id)
                    if id >= 0:
                        self.opt.gpu_ids.append(id)
        else:
            self.opt.gpu_ids = []
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
