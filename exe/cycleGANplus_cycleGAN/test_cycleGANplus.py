################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################


import time
import os
from xifontgan.options.XItest_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from xifontgan.data.XIdata_loader import CreateDataLoader
from xifontgan.models.XImodels import create_model
from xifontgan.util.XIvisualizer import Visualizer
from pdb import set_trace as st
from xifontgan.util import XIhtml

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle
opt.stack = True
opt.use_dropout = False
opt.use_dropout1 = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch+'+'+opt.which_epoch1))
webpage = XIhtml.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch + '+' + opt.which_epoch1))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path,
                           test_gt_dir=test_gt_dir)

webpage.save()