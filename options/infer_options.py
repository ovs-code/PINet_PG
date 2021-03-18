import argparse
import torch

class InferOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # configure model
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--which_model_netG', type=str, default='PInet', help='selects model to use for netG')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        self.parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.parser.add_argument('--P_input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--BP_input_nc', type=int, default=18, help='# of input image channels')

        self.parser.add_argument('--with_D_PP', type=int, default=1, help='use D to judge P and P is pair or not')
        self.parser.add_argument('--with_D_PB', type=int, default=1, help='use D to judge P and B is pair or not')

        self.parser.add_argument('--G_n_downsampling', type=int, default=2, help='down-sampling blocks for generator')
        self.parser.add_argument('--D_n_downsampling', type=int, default=2, help='down-sampling blocks for discriminator')

        self.parser.add_argument('--dataset_mode', type=str, default='keypoint', help='chooses how datasets are loaded. [unaligned | aligned | single | keypoint]')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')

        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        # interesting arguments
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument("--pose_estimator", default='pose_estimator.h5', help='Pretrained model for cao pose estimator')



        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.model = 'PInet'
        self.opt.isTrain = False

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        return self.opt
