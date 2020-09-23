#=====================================
# XI-fontGAN
# from MC-GAN
# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#=====================================

if __package__ is None:
    import os
    import sys
    import time
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]
    sys.path.insert(0, project_root)
    from xifontgan.options.XItrain_options import TrainOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TrainOptions().parse()
    from xifontgan.models.XImodels import create_model
    from xifontgan.util.XIvisualizer import Visualizer
    from xifontgan.data.XIdata_loader import CreateDataLoader
    import configparser
else:
    import time
    from ...options.XItrain_options import TrainOptions
    # set CUDA_VISIBLE_DEVICES before import torch
    opt = TrainOptions().parse()
    from ...models.XImodels import create_model
    from ...util.XIvisualizer import Visualizer
    from ...data.XIdata_loader import CreateDataLoader
    import configparser


# read configuration from files
configParser = configparser.RawConfigParser()
configFilePath = r''

data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data()
dataset_size = len(data_loader)

model = create_model(opt)
visualizer = Visualizer(opt)

# get the first epoch if resuming
first_epoch = opt.which_epoch + 1

dataset_training_size = dataset_size - dataset_size % opt.batchSize
print('#training images = %d / %d' % (dataset_training_size,
                                      dataset_size))
print('LOADING from %s' % opt.which_epoch)
total_steps = (first_epoch - 1) * dataset_training_size
last_display_step = total_steps
last_print_step = total_steps
last_save_step = total_steps

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
        epoch_iter = total_steps - dataset_training_size * (epoch - 1)
        model.set_LSTM()
        # print('data %d' % i)
        for j in range(remainnum):
            # print('Time: %d ' % j)
            model.set_input(data, j)
            model.forward()
            model.optimize_parameters()

        if total_steps > (last_display_step + opt.display_freq - 1):
            visualizer.display_current_results(
                model.get_current_visuals(0), epoch)
            last_display_step = total_steps

        if total_steps > (last_print_step + opt.print_freq - 1):
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors,
                                            t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(
                    epoch_iter) / dataset_training_size, opt, errors)
            last_print_step = total_steps

        # if total_steps > (last_save_step + opt.save_latest_freq - 1):
        #     print(
        #         'saving the latest model (epoch %d, total_steps %d)' %
        #         (epoch, total_steps))
        #     model.save('latest')
        #     last_save_step = total_steps

        if opt.batchSize==1:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            print('process image... %s' % img_path, end =" ")
            scores = visualizer.eval_current_result(visuals)
            print("L1: %s" % (scores[0]), end =" ")
            print("ssim: %s" % (scores[1]), end =" ")
            print("MSE: %s" % (scores[2]))
            l1_score += scores[0]
            ssim_score += scores[1]
            mse_score += scores[2]

    if opt.batchSize==1:
        print("Final scores for %d images:" % (i + 1),
              "L1: %s" % (l1_score / (i + 1)),
              "ssim: %s" % (ssim_score / (i + 1)),
              "MSE: %s" % (mse_score / (i + 1)))

    print('saving the latest model (epoch %s, total_steps %d)' % (
        model.epoch_str, total_steps))
    model.save(latest=True)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %s, iters %d' %
              (model.epoch_str, total_steps))
        model.save()

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
