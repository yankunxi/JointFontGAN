#=====================================
# MC-GAN
# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#=====================================

import time
from options.XItrain_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from xifontgan.models import create_model
from xifontgan.util import Visualizer
from xifontgan.data import CreateDataLoader
import configparser

# read configuration from files
configParser = configparser.RawConfigParser()
configFilePath = r''

data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
print('LOADING from %s' % opt.which_epoch)


model = create_model(opt)
visualizer = Visualizer(opt)

# get the first epoch if resuming
if opt.which_epoch == 'latest':
    first_epoch = 0
else:
    first_epoch = int(opt.which_epoch)

total_steps = first_epoch * dataset_size

for epoch in range(first_epoch + 1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()
        
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
