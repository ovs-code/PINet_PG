import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os.path
import random
import pandas as pd
import numpy as np
import torch

from . import background

class KeyDataset(data.Dataset):
    def __init__(self):
        super(KeyDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # "/home/zjs/Pose_Transfer/fashion_data", opt.phase) #person image
        self.dir_P = os.path.join(opt.dataroot, opt.phase)
        # "/home/zjs/Pose_Transfer/fashion_data", opt.phase + 'K') #keypoints
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')
        # self.dir_S = os.path.join(opt.dataroot, opt.phase + 'S') # person segmentation
        # spl2 12 full #spl3 10 full
        self.dir_SL = os.path.join(opt.dataroot, opt.phase + 'SPL2')
        self.class_num = 12

        self.get_datapairs(os.path.join(
            opt.dataroot, f'fasion-resize-pairs-{opt.phase}.csv'))
        self.transform = self.get_transform(opt)

        if opt.use_bg_augmentation or opt.use_bg_augmentation_both:
            self.backgrounds = background.load_backgrounds('data/backgrounds')

    def get_datapairs(self, pairLst):
        '''
        get the data pairs from csv file
        '''
        pairs_file = pd.read_csv(pairLst)
        self.size = len(pairs_file)
        self.pairs = []
        print('loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def get_transform(self, opt):
        transform_list = []
        if opt.resize_or_crop == 'resize_and_crop':
            osize = [opt.loadSize, opt.loadSize]
            transform_list.append(transforms.Scale(osize, Image.BICUBIC))
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        elif opt.resize_or_crop == 'scale_width':
            transform_list.append(transforms.Lambda(
                lambda img: __scale_width(img, opt.fineSize)))
        elif opt.resize_or_crop == 'scale_width_and_crop':
            transform_list.append(transforms.Lambda(
                lambda img: __scale_width(img, opt.loadSize)))
            transform_list.append(transforms.RandomCrop(opt.fineSize))

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):

        # crop all the input to 256*176

        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        regions = (40, 0, 216, 256)  # crop image to 256*176

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name)  # person 1
        # keypoints of person 1
        KP1_path = os.path.join(self.dir_K, P1_name + '.npy')

        P2_path = os.path.join(self.dir_P, P2_name)  # person 2
        # keypoints of person 2
        KP2_path = os.path.join(self.dir_K, P2_name + '.npy')

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        KP1_img = np.load(KP1_path)  # [:,:-80,:]  #h, w, c
        KP2_img = np.load(KP2_path)  # [:,:-80,:]
        if np.array(P1_img).shape[1] == 256:
            P1_img = P1_img.crop(regions)
            P2_img = P2_img.crop(regions)
        if KP1_img.shape[1] == 256:
            KP1_img = KP1_img[:, :-80, :]
            KP2_img = KP2_img[:, :-80, :]

        SPL1_path = os.path.join(self.dir_SL, P1_name[:-4]+'.png')
        SPL2_path = os.path.join(self.dir_SL, P2_name[:-4]+'.png')
        SPL1_img = Image.open(SPL1_path)
        if SPL1_img.size[0] == 256:
            SPL1_img = SPL1_img.crop(regions)
        SPL2_img = Image.open(SPL2_path)
        if SPL2_img.size[0] == 256:
            SPL2_img = SPL2_img.crop(regions)

        if self.opt.phase == 'train' and self.opt.brightness_augmentation:
            brightness_factor = random.uniform(0.5, 1)
            P1_img = transforms.functional.adjust_brightness(P1_img, brightness_factor)
            P2_img = transforms.functional.adjust_brightness(P2_img, brightness_factor)
        if self.opt.phase == 'train' and self.opt.use_flip:
            flip_random = random.uniform(0, 1)

            if flip_random > 0.5:
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                KP1_img = np.array(KP1_img[:, ::-1, :])  # flip
                KP2_img = np.array(KP2_img[:, ::-1, :])  # flip

                SPL1_img = SPL1_img.transpose(Image.FLIP_LEFT_RIGHT)
                SPL2_img = SPL2_img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.opt.phase == 'train' and self.opt.use_topbottom_flip:
            flip_random = random.uniform(0, 1)

            if flip_random > 0.9:
                P2_img = P2_img.transpose(Image.FLIP_TOP_BOTTOM)

                KP2_img = np.array(KP2_img[::-1, :, :])  # flip

                SPL2_img = SPL2_img.transpose(Image.FLIP_TOP_BOTTOM)



        SPL1_img = np.array(SPL1_img)
        SPL2_img = np.array(SPL2_img)

        if self.opt.phase == 'train' and self.opt.use_bg_augmentation:
            # insert background into source image
            bg = random.choice(self.backgrounds)
            P1_img = background.background_swap(np.array(P1_img), SPL1_img, bg)
            # remove background from target image
            P2_img = background.remove_background(np.array(P2_img), SPL2_img)
        elif self.opt.phase == 'train' and self.opt.use_bg_augmentation_both:
            # insert background into source image
            bg = random.choice(self.backgrounds)
            P1_img = background.background_swap(np.array(P1_img), SPL1_img, bg)
            # remove background from target image
            P2_img = background.background_swap(np.array(P2_img), SPL2_img, bg)
        elif self.opt.remove_background:
            white = np.ones((256, 176), dtype=np.uint8) * 255
            P1_img = background.remove_background(np.array(P1_img), SPL1_img)
            P2_img = background.remove_background(np.array(P2_img), SPL2_img)


        KP1 = torch.from_numpy(KP1_img).float()  # h, w, c
        KP1 = KP1.transpose(2, 0)  # c,w,h
        KP1 = KP1.transpose(2, 1)  # c,h,w

        KP2 = torch.from_numpy(KP2_img).float()
        KP2 = KP2.transpose(2, 0)  # c,w,h
        KP2 = KP2.transpose(2, 1)  # c,h,w

        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)
        SPL1_img = np.expand_dims(SPL1_img, 0)  # 1*256*176
        SPL2_img = np.expand_dims(SPL2_img, 0)
        # reduce number of parsing classes
        # dress -> upper clothes
        SPL1_img = np.where(SPL1_img==3, 2, SPL1_img)
        SPL2_img = np.where(SPL2_img==3, 2, SPL2_img)
        # skirt -> pants
        SPL1_img = np.where(SPL1_img==6, 4, SPL1_img)
        SPL2_img = np.where(SPL2_img==6, 4, SPL2_img)

        _, h, w = SPL2_img.shape
       # print(SPL2_img.shape,SPL1_img.shape)
        num_class = self.class_num
        tmp = torch.from_numpy(SPL2_img).view(-1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([h, w, num_class])
        # print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(2, 0, 1)

        tmp = torch.from_numpy(SPL1_img).view(-1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL1_onehot = ones.view([h, w, num_class])
        # print(SPL2_onehot.shape)
        SPL1_onehot = SPL1_onehot.permute(2, 0, 1)

        SPL1 = torch.from_numpy(SPL1_img).long()
        SPL2 = torch.from_numpy(SPL2_img).long()
        return dict(
            P1=P1, KP1=KP1, P2=P2, KP2=KP2, SPL1=SPL1, SPL2=SPL2,
            SPL1_onehot=SPL1_onehot, SPL2_onehot=SPL2_onehot,
            P1_path=P1_name, P2_path=P2_name
        )

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase in ('test', 'val'):
            return self.size

    def name(self):
        return 'KeyDataset'
