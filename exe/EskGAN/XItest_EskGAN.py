#=====================================
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
#=====================================
import os
import sys
import shutil
import time
import configparser
from pdb import set_trace as st
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
else:
    from ...options.XItest_options import TestOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TestOptions().parse()
    from ...models.XImodels import create_model
    from ...util import XIdecide
    from ...util.XIvisualizer import Visualizer
    from ...util import XIhtml
    from ...data.XIdata_loader import CreateDataLoader


opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
# create website
web_dir, webpage = XIdecide.generate_results_dirs(opt,
                                                  model.epoch_str)
test_dict = XIdecide.get_test_dict(opt)

# test
l1_score = 0
ssim_score = 0
mse_score = 0

#test data saving
# if opt.phase == 'train':
if opt.phase.startswith('test'):
    if os.path.isfile(test_dict):
        test_gt_dir, test_gt_compress = XIdecide.get_test_gt(opt)
    else:
        test_gt_dir = ''

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    scores = visualizer.eval_current_result(visuals,
                                            op='fake_B',
                                            gt='real_B')
    print("L1: %3.7f" % (scores[0]), end="\t")
    print("ssim: %3.7f" % (scores[1]), end="\t")
    print("MSE: %3.7f" % (scores[2]), end="\t")
    print('for image #%d ...%s' % (i + 1, img_path[0].rpartition(
        "English")[-1]))
    l1_score += scores[0]
    ssim_score += scores[1]
    mse_score += scores[2]
    visualizer.save_images(webpage, visuals, img_path,
                           test_gt_dir=test_gt_dir)

print("Final scores for %d images:" % (i + 1),
      "L1: %s" % (l1_score / (i + 1)),
      "ssim: %s" % (ssim_score / (i + 1)),
      "MSE: %s" % (mse_score / (i + 1)))
webpage.save()

if opt.zip_results:
    print('opt.zip_results %s' % opt.zip_results)
    web_zip = os.path.join('.', opt.results_dir, opt.experiment_dir,
                           '%s_%s' % (opt.phase, model.epoch_str))
    if os.path.isfile(test_dict):
        if not os.path.isfile(test_gt_compress + '/gt.zip'):
            shutil.make_archive(test_gt_compress + '/gt', 'zip',
                                test_gt_dir + '/gt')
            print('gt ZIP saved to %s/gt.zip', test_gt_compress)
        else:
            print('gt ZIP file already exists.')
        if not os.path.isfile(test_gt_compress + '/Egt.zip'):
            shutil.make_archive(test_gt_compress + '/Egt', 'zip',
                                test_gt_dir + '/Egt')
            print('Egt ZIP saved to %s/Egt.zip', test_gt_compress)
        else:
            print('Egt ZIP file already exists.')
        if not os.path.isfile(test_gt_compress + '/gt_sk.zip'):
            shutil.make_archive(test_gt_compress + '/gt_sk', 'zip',
                                test_gt_dir + '/gt_sk')
            print('gt_sk ZIP saved to %s/gt_sk.zip', test_gt_compress)
        else:
            print('gt_sk ZIP file already exists.')
        if not os.path.isfile(test_gt_compress + '/Egt_sk.zip'):
            shutil.make_archive(test_gt_compress + '/Egt_sk', 'zip',
                                test_gt_dir + '/Egt_sk')
            print('Egt_sk ZIP saved to %s/Egt_sk.zip',
                  test_gt_compress)
        else:
            print('Egt_sk ZIP file already exists.')
    else:
        print('Test dictionary not existing.')
    if not os.path.isfile(web_zip + '.zip'):
        shutil.make_archive(web_zip, 'zip', web_dir)
        print('ZIP file saved to %s.zip' % web_zip)
    else:
        print('ZIP file already exists.')
else:
    print('Not zipping per request.')

