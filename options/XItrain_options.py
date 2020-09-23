################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################


from .XIbase_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=10000, help='frequency of '
                                                     'saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # self.parser.add_argument('--nepoch', type=int, default=200, help='# of epochs to train the model')
        self.parser.add_argument('--gamma', type=int, default=0.0001, help='how much to decrease learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for G1_L1')
        self.parser.add_argument('--lambda_C', type=float, default=0.0, help='weight for G_L1')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer_path that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.experiment_dir]/web/')
        self.parser.add_argument('--noisy_disc', action='store_true', help='add noise to the discriminator target labels')

        self.parser.add_argument('--ntest', type=int,
                                 default=float("inf"),
                                 help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str,
                                 default='results',
                                 help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float,
                                 default=1.0,
                                 help='aspect ratio of result images')
        self.parser.add_argument('--how_many', type=int,
                                 default=10000,
                                 help='how many test images to run')

        self.isTrain = True
