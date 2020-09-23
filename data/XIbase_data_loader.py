#=============================
# From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#=============================
import os

class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt

        if self.opt.use_auxiliary:
            self.dataroot = os.path.join(self.opt.auxiliary_root,
                                         self.opt.data_relative,
                                         self.opt.dataset)
            self.baseroot = os.path.join(self.opt.auxiliary_root,
                                         self.opt.data_relative,
                                         self.opt.base_root)
        else:
            self.dataroot = os.path.join(self.opt.everything_root,
                                         self.opt.data_relative,
                                         self.opt.dataset)
            self.dataroot = os.path.join(self.opt.everything_root,
                                         self.opt.data_relative,
                                         self.opt.base_root)
        self.fineSize = opt.fineSize
        pass

    def load_data():
        return None
