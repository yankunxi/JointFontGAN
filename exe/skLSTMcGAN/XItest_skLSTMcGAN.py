#=====================================
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
#=====================================

import os
import sys
import shutil
import time
if __package__ is None:
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]
    sys.path.insert(0, project_root)
    from xifontgan.options.XItrain_options import TrainOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TrainOptions().parse()
    from xifontgan.models.XImodels import create_model
    from xifontgan.util.XIvisualizer import Visualizer
    from xifontgan.util import XIhtml
    from xifontgan.data.XIdata_loader import CreateDataLoader
    import configparser
    from pdb import set_trace as st
else:
    from ...options.XItrain_options import TrainOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TrainOptions().parse()
    from ...models.XImodels import create_model
    from ...util.XIvisualizer import Visualizer
    from ...util import XIhtml
    from ...data.XIdata_loader import CreateDataLoader
    import configparser
    from pdb import set_trace as st


opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

n_rgb = 3 if opt.rgb else 1
if opt.str_input:
    remainnum = len(opt.str_input)
    blanknum = int(opt.input_nc / n_rgb) - remainnum
else:
    blanknum = int(opt.blanks * opt.input_nc / n_rgb)
    remainnum = int(opt.input_nc / n_rgb) - blanknum

# create website
if opt.use_auxiliary:
    web_dir = os.path.join(opt.auxiliary_root, opt.project_relative,
                           opt.results_dir, opt.experiment_dir,
                           '%s_%s' % (opt.phase, model.epoch_str))
else:
    web_dir = os.path.join('.', opt.results_dir, opt.experiment_dir,
                           '%s_%s' % (opt.phase, model.epoch_str))
webpage = XIhtml.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = '
                             '%s' % (opt.experiment_dir, opt.phase,
                                     model.epoch_str))

# test
l1_score = 0
ssim_score = 0
mse_score = 0
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_LSTM()

    for j in range(remainnum):
        # print('Time: %d ' % j)
        model.set_input(data, j)
        model.test()
        if opt.str_output:
            for k in range(len(opt.str_output)):
                model.set_test_input(data, k, j)
                model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        visualizer.save_images(webpage, visuals, img_path, tag=str(j))


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


print("Final scores for %d images:" % (i + 1),
      "L1: %s" % (l1_score / (i + 1)),
      "ssim: %s" % (ssim_score / (i + 1)),
      "MSE: %s" % (mse_score / (i + 1)))
webpage.save()
if opt.use_auxiliary:
    web_zip = os.path.join('.', opt.results_dir, opt.experiment_dir,
                           '%s_%s' % (opt.phase, model.epoch_str))
shutil.make_archive(web_zip, 'zip', web_dir)