#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================

import numpy as np
import os
import ntpath
import time
from . import XIutil
from . import XIhtml
import shutil
from scipy import misc
from skimage import data, img_as_float
import skimage.measure
import skvideo.io


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.stack_result = opt.stack_result
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        if self.opt.use_auxiliary:
            expr_dir = os.path.join(self.opt.auxiliary_root,
                                    self.opt.project_relative,
                                    self.opt.checkpoints_dir,
                                    self.opt.experiment_dir)
        else:
            expr_dir = os.path.join(".", self.opt.checkpoints_dir,
                                    self.opt.experiment_dir)
        self.name = os.path.dirname(expr_dir)
        self.rgb = opt.rgb or opt.rgb_out
        self.video_dir = []
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom()
        if self.use_html:
            self.web_dir = os.path.join(expr_dir, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            if self.opt.use_auxiliary:
                print('create auxiliary web directory %s...' % self.web_dir)
            else:
                print('create web directory %s...' % self.web_dir)
            XIutil.mkdirs([self.web_dir, self.img_dir])

    def eval_current_result(self, visuals, op='fake_B',
                            gt='real_B'):
        op = visuals[op].copy()
        gt = visuals[gt].copy()
        # print(op.max(), gt.max())
        ssim_score = skimage.measure.compare_ssim(op, gt,
                                                  data_range=gt.max() - gt.min())
        float_type = np.result_type(op.dtype, gt.dtype, np.float32)
        op = np.asarray(op, dtype=float_type)
        gt = np.asarray(gt, dtype=float_type)
        op /= gt.max()
        gt /= gt.max()
        # print(op.max(), gt.max())
        # print(op)
        l1_score = np.mean(np.absolute(op - gt))
        mse_score = np.mean((op - gt)**2)
        return l1_score, ssim_score, mse_score

    # |visuals|: dictionary of images to display or save
    # TO BE FINISHED
    def display_current_results(self, visuals, epoch_str, last=False):
        if self.stack_result:
            image_numpy_ = []
            image_numpy_e = []
            for label, image_numpy in visuals.items():
                if label[0] == 'E':
                    image_numpy_e = image_numpy_e + list(
                        image_numpy)
                else:
                    image_numpy_ = image_numpy_ + list(
                        image_numpy)
            image_numpy_e = np.asanyarray(image_numpy_e)
            image_numpy_ = np.asanyarray(image_numpy_)
        if self.display_id > 0: # show images in the browser
            idx = 1
            if self.stack_result:
                if image_numpy_e.shape[0]:
                    self.vis.image(image_numpy_e.transpose([2, 0,
                                                             1]),
                                   opts=dict(title='ExtendedStack'),
                                   win=self.display_id + idx)
                    idx += 1
                if image_numpy_.shape[0]:
                    self.vis.image(image_numpy_.transpose([2, 0, 1]),
                                   opts=dict(title='Stack'),
                                   win=self.display_id + idx)
                    idx += 1
            else:
                for label, image_numpy in visuals.items():
                    # image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2, 0, 1]),
                                   opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html:  # save images to a html file
            if self.stack_result:
                if image_numpy_e.shape[0]:
                    # print("numpy_e.shape is %s" % image_numpy_e.shape)
                    img_path = os.path.join(self.img_dir,
                                            'epoch%s_%s.png' % (epoch_str, 'ExtendedStack'))
                    XIutil.save_image(image_numpy_e, img_path,
                                      self.rgb)
                    img_path_ = os.path.join(self.img_dir,
                                             'latest_ExtendedStack.png')
                    XIutil.save_image(image_numpy_e, img_path_,
                                      self.rgb)
                if image_numpy_.shape[0]:
                    img_path = os.path.join(self.img_dir,
                                            'epoch%s_%s.png' % (epoch_str, 'Stack'))
                    XIutil.save_image(image_numpy_, img_path,
                                      self.rgb)
                    img_path_ = os.path.join(self.img_dir,
                                             'latest_Stack.png')
                    XIutil.save_image(image_numpy_, img_path_,
                                      self.rgb)
            else:
                for label, image_numpy in visuals.items():
                    img_path = os.path.join(self.img_dir,
                                            'epoch%s_%s.png' % (epoch_str, label))
                    XIutil.save_image(image_numpy, img_path, self.rgb)
                    img_path_ = os.path.join(self.img_dir,
                                             'latest_%s.png' % label)
                    XIutil.save_image(image_numpy, img_path_,
                                      self.rgb)

            # # update website
            # webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            # for n in range(epoch, 0, -1):
            #     webpage.add_header('epoch [%d]' % n)
            #     ims = []
            #     txts = []
            #     links = []
            #
            #     for label, image_numpy in visuals.items():
            #         img_path = 'epoch%.3d_%s.png' % (n, label)
            #         ims.append(img_path)
            #         txts.append(label)
            #         links.append(img_path)
            #     webpage.add_images(ims, txts, links, width=self.win_size)
            # webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, tag="",
                    test_gt_dir=''):
        image_dir = webpage.get_image_dir()
        print("save to:", image_dir)
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0] + tag

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        if self.stack_result:
            image_numpy_ = []
            image_numpy_e = []
            for label, image_numpy in visuals.items():
                if label[0] == 'E':
                    image_numpy_e = image_numpy_e + list(
                        image_numpy)
                else:
                    image_numpy_ = image_numpy_ + list(
                        image_numpy)
            image_numpy_e = np.asanyarray(image_numpy_e)
            image_numpy_ = np.asanyarray(image_numpy_)

            if sum(image_numpy_e.shape):
                image_name = '%s_%s.png' % (name, 'ExtendedStack')
                save_path = os.path.join(image_dir, image_name)
                XIutil.save_image(image_numpy_e, save_path, self.rgb)
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
            if sum(image_numpy_.shape):
                image_name = '%s_%s.png' % (name, 'Stack')
                save_path = os.path.join(image_dir, image_name)
                XIutil.save_image(image_numpy_, save_path, self.rgb)
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        else:
            for label, image_numpy in visuals.items():
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(image_dir, image_name)
                if test_gt_dir:
                    # if label == 'redbox':
                    #     XIutil.mkdirs(test_gt_dir + '/redbox')
                    #     save_path = os.path.join(test_gt_dir,
                    #                              'redbox', image_name)
                    temp = test_gt_dir
                    if label == 'real_A' or label == 'real_B':
                        XIutil.mkdirs(test_gt_dir + '/gt')
                        save_path = os.path.join(test_gt_dir, 'gt',
                                                 image_name)
                    elif label == 'Ereal_A' or label == 'Ereal_B':
                        XIutil.mkdirs(test_gt_dir + '/Egt')
                        save_path = os.path.join(test_gt_dir, 'Egt',
                                                 image_name)
                    elif label == 'real_Ask1' or label == 'real_Bsk1':
                        XIutil.mkdirs(test_gt_dir + '/gt_sk')
                        save_path = os.path.join(test_gt_dir, 'gt_sk',
                                                 image_name)
                    elif label == 'Ereal_Ask1' or label == 'Ereal_Bsk1':
                        XIutil.mkdirs(test_gt_dir + '/Egt_sk')
                        save_path = os.path.join(test_gt_dir,
                                                 'Egt_sk', image_name)
                    else:
                        XIutil.save_image(image_numpy, save_path,
                                          self.rgb)
                        temp = '+'
                    if not temp == '+':
                        XIutil.save_image(image_numpy, save_path,
                                          self.rgb)
                else:
                    XIutil.save_image(image_numpy, save_path,
                                      self.rgb)
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
        self.video_dir.extend([image_dir])
        
    def copy_images(self, image_paths, image_path, dest_path_A, dest_path_B, target_path, observed):
        name = os.path.splitext(image_path)[0]

        saved_path = os.path.join(image_paths, image_path)
        im_ = misc.imread(saved_path)
        n_ch = im_.shape[1]/im_.shape[0]
        target_size = im_.shape[0]
        for obs in observed:
            im_[:,target_size*obs:(obs+1)*target_size,:] = 255 
        for obs in observed:
        	image_name = '%s_%s.png' % (name, obs)
        	im_path = os.path.join(dest_path_A, image_name)
        	misc.imsave(im_path, im_)
            
        target_ims = os.listdir(target_path)
        for im in target_ims:
            obs = im.split('.png')[0].split('_')[-1]
            image_name = '%s_%s.png' % (name, obs)
            shutil.copyfile(os.path.join(target_path,im), os.path.join(dest_path_B,image_name))
        
    def save_video(self,video_path):
        outputdata = []
        os.system("mkdir ~/tmp_ffmpeg")
        os.system("mkdir ~/tmp_ffmpeg/fake_B")
        os.system("mkdir ~/tmp_ffmpeg/real_A")
        os.system("rm ~/tmp_ffmpeg/*")
        video =None
        inpout=['fake_B','real_A']
        for end_ in inpout:
            i=1
            for img_dir in self.video_dir:
                imgs = sorted(os.listdir(img_dir))
                num_imgs =len(imgs)/3
                for im in imgs:
                        if im.endswith('_%s.png'%end_):
                            img = misc.imread(img_dir+'/'+im)
                            im_size = img.shape[0]
                            epoch = im.split('_')[0]
                            print(epoch, i)
                            # shutil.copyfile(os.path.join(img_dir,im),"~/tmp_ffmpeg/im%04d.png"%i)
                            os.system("cp %s ~/tmp_ffmpeg/%s/im%04d.png 2>&1|tee ~/tmp_ffmpeg/log.txt"%(os.path.join(img_dir.replace("&","\&"),im),end_,i))
                            i+=1
        os.system("ffmpeg -r 6 -f image2 -s %dx%d -i ~/tmp_ffmpeg/fake_B/im%%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p %s/test_fake_B.mp4 2>&1|tee ~/tmp_ffmpeg/log2_fake_B.txt"%(img.shape[0],img.shape[1],video_path.replace("&","\&")))
        os.system("ffmpeg -r 6 -f image2 -s %dx%d -i ~/tmp_ffmpeg/real_A/im%%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p %s/test_real_A.mp4 2>&1|tee ~/tmp_ffmpeg/log2_real_A.txt"%(img.shape[0],img.shape[1],video_path.replace("&","\&")))

