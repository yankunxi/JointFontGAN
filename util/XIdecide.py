#=====================================
# JointFontGAN
# By Yankun Xi
#=====================================

import os
from . import XIhtml


def generate_results_dirs(opt, model_epoch_str):
    if opt.use_auxiliary:
        results_dir_ = os.path.join(opt.auxiliary_root,
                                    opt.project_relative,
                                    opt.results_dir,
                                    opt.experiment_dir)
    else:
        results_dir_ = os.path.join('.', opt.results_dir,
                                    opt.experiment_dir,)
    if opt.str_input:
        web_dir = os.path.join(results_dir_, '%s_%s_%s' % (
            opt.phase, model_epoch_str, opt.str_input), '_')
    else:
        web_dir = os.path.join(results_dir_, '%s_%s' % (opt.phase,
                                                        model_epoch_str))
    webpage = XIhtml.HTML(web_dir, 'Experiment = %s, Phase = %s, '
                                   'Epoch = %s'
                          % (opt.experiment_dir, opt.phase,
                             model_epoch_str))
    return web_dir, webpage


def get_test_dict(opt):
    if opt.use_auxiliary:
        test_dict = os.path.join(opt.auxiliary_root,
                                 opt.data_relative,
                                 opt.dataset) + '/test_dict/dict.pkl'
    else:
        test_dict = os.path.join(opt.everything_root,
                                 opt.data_relative,
                                 opt.dataset) + '/test_dict/dict.pkl'
    return test_dict


def get_test_gt(opt):
    if opt.use_auxiliary:
        results_dataset_ = os.path.join(opt.auxiliary_root,
                                        opt.project_relative,
                                        opt.results_dir, opt.dataset)
    else:
        results_dataset_ = os.path.join('.', opt.results_dir,
                                        opt.dataset)
    if opt.str_input:
        test_gt_dir = os.path.join(results_dataset_,
                                   'string_%s' % opt.str_input)
        test_gt_compress = os.path.join('.', opt.results_dir,
                                        opt.dataset,
                                        'string_%s' % opt.str_input)
    else:
        test_gt_dir = os.path.join(results_dataset_,
                                   'blanks_%s' % opt.blanks)
        test_gt_compress = os.path.join('.', opt.results_dir,
                                        opt.dataset,
                                        'blanks_%s' % opt.blanks)
    if os.path.isdir(test_gt_dir + '/gt'):
        if os.path.isdir(test_gt_dir + '/Egt'):
            if os.path.isdir(test_gt_dir + '/gt_sk'):
                if os.path.isdir(test_gt_dir + '/Egt_sk'):
                    test_gt_dir = '+'
    return test_gt_dir, test_gt_compress





