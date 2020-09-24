#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================

from .XIbase_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--how_many', type=int, default=10000, help='how many test images to run')
        self.isTrain = False
