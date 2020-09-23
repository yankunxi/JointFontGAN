################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################


import os
import sys
import shutil
import time
if __package__ is None:
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]
    sys.path.insert(0, project_root)
    from xifontgan.options.XItest_options import TestOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TestOptions().parse()
    from xifontgan.models.XImodels import create_model
    from xifontgan.util import XIdecide
    from xifontgan.util.XIvisualizer import Visualizer
    from xifontgan.util import XIhtml
    from xifontgan.data.XIdata_loader import CreateDataLoader
    import configparser
    from pdb import set_trace as st
else:
    from ...options.XItest_options import TestOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TestOptions().parse()
    from ...models.XImodels import create_model
    from ...util import XIdecide
    from ...util.XIvisualizer import Visualizer
    from ...util import XIhtml
    from ...data.XIdata_loader import CreateDataLoader
    import configparser
    from pdb import set_trace as st

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  # test code only supports batchSize=1
opt.serial_batches = True # no shuffle
opt.stack = True
opt.use_dropout = False
opt.use_dropout1 = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir, webpage = XIdecide.generate_results_dirs(opt,
                                                  model.epoch_str)
test_dict = XIdecide.get_test_dict(opt)

# test
l1_score = 0
ssim_score = 0
mse_score = 0

# test data saving
# if opt.phase == 'train':
if opt.phase.startswith('test'):
    if os.path.isfile(test_dict):
        test_gt_dir = os.path.join(opt.auxiliary_root,
                                   opt.project_relative,
                                   opt.results_dir,
                                   opt.dataset,
                                   'blanks_%s' % opt.blanks)
        test_gt_compress = os.path.join('.', opt.results_dir,
                                        opt.dataset,
                                        'blanks_%s' % opt.blanks)
        if os.path.isdir(test_gt_dir + '/gt'):
            if os.path.isdir(test_gt_dir + '/redbox'):
                test_gt_dir = ''
        # print('test & save')
    else:
        test_gt_dir = ''

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    # scores = visualizer.eval_current_result(visuals,
    #                                         op='fake_B',
    #                                         gt='real_B')
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path,
                           test_gt_dir=test_gt_dir)
print("Final scores for %d images:" % (i + 1),
      "L1: %s" % (l1_score / (i + 1)),
      "ssim: %s" % (ssim_score / (i + 1)),
      "MSE: %s" % (mse_score / (i + 1)))
webpage.save()
