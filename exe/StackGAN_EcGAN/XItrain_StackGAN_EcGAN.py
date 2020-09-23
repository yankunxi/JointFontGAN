################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################

import os
import sys
import shutil
import time
import configparser
if __package__ is None:
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]
    sys.path.insert(0, project_root)
    from xifontgan.options.XItrain_options import TrainOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TrainOptions().parse()
    from xifontgan.models.XImodels import create_model
    from xifontgan.util.XIvisualizer import Visualizer
    from xifontgan.data.XIdata_loader import CreateDataLoader
else:
    from ...options.XItrain_options import TrainOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TrainOptions().parse()
    from ...models.XImodels import create_model
    from ...util.XIvisualizer import Visualizer
    from ...data.XIdata_loader import CreateDataLoader

opt.stack = True
# read configuration from files
configParser = configparser.RawConfigParser()
configFilePath = r''

data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
opt.use_dropout = True
opt.use_dropout1 = True
model = create_model(opt)
visualizer = Visualizer(opt)

# get the first epoch if resuming
first_epoch = opt.which_epoch + 1

dataset_training_size = dataset_size - dataset_size % opt.batchSize
print('#training images = %d / %d' % (dataset_training_size,
                                      dataset_size))
print('LOADING from %s' % opt.which_epoch)
total_steps = (first_epoch - 1) * dataset_training_size
last_display_step = total_steps-1000
last_print_step = total_steps-1000
last_save_step = total_steps-1000

n_rgb = 3 if opt.rgb else 1
blanknum = int(opt.blanks * opt.input_nc / n_rgb)
remainnum = int(opt.input_nc / n_rgb) - blanknum

l1_score = 0
ssim_score = 0
mse_score = 0

for epoch in range(first_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    model.next_epoch()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - (dataset_size - dataset_size %
                                    opt.batchSize) * (epoch - 1)
        model.set_input(data)
        if not opt.no_Style2Glyph:
            model.optimize_parameters_Stacked(epoch)
        else:
            model.optimize_parameters(epoch)

        if total_steps > (
                last_display_step + opt.display_freq - 1):
            visualizer.display_current_results(
                model.get_current_visuals(), model.epoch_str)

            last_display_step = total_steps

        if total_steps > (last_print_step + opt.print_freq - 1):
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter,
                                            errors,
                                            t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(
                    epoch_iter) / dataset_training_size, opt,
                                               errors)
            last_print_step = total_steps

        # if total_steps > (last_save_step + opt.save_latest_freq - 1):
        #     print(
        #         'saving the latest model (epoch %d, total_steps %d)' %
        #         (epoch, total_steps))
        #     model.save('latest')
        #     last_save_step = total_steps

        if opt.batchSize == 1:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            print('process image... %s' % img_path, end=" ")
            scores = visualizer.eval_current_result(visuals)
            print("L1: %s" % (scores[0]), end=" ")
            print("ssim: %s" % (scores[1]), end=" ")
            print("MSE: %s" % (scores[2]))
            l1_score += scores[0]
            ssim_score += scores[1]
            mse_score += scores[2]

    if opt.batchSize == 1:
        print("Final scores for %d images:" % (i + 1),
              "L1: %s" % (l1_score / (i + 1)),
              "ssim: %s" % (ssim_score / (i + 1)),
              "MSE: %s" % (mse_score / (i + 1)))

    print('saving the latest model (epoch %s, total_steps %d)' % (
        model.epoch_str, total_steps))
    model.save(latest=True)

    if epoch % opt.save_epoch_freq == 0:
        print(
            'saving the model at the end of epoch %s, iters %d' %
            (model.epoch_str, total_steps))
        model.save()

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay,
           time.time() - epoch_start_time))
