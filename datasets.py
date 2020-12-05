# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import random
import math
#############################################################################################################
# Channel_Dataset: It is used to get image pairs, which have the same sketch and different contents
# Parameters
#         ----------
#         domain_num: The number of augmented samples for each original one
# -----------------------------------------------------------------------------------------------------------
class Channel_Dataset(Dataset):
    def __init__(self, root, transform=None, targte_transform=None, domain_num=6):
        super(Channel_Dataset, self).__init__()
        self.image_dir = root
        self.samples = []
        self.img_label = []
        self.transform = transform
        self.targte_transform = targte_transform
        self.class_num = len(os.listdir(self.image_dir))  # the number of the class
        self.domain_num = domain_num
        print('self.class_num = %s' % self.class_num)
        dirs = os.listdir(self.image_dir)
        for dir in dirs:
            fdir = os.path.join(self.image_dir, dir)
            files = os.listdir(fdir)
            for file in files:
                self.img_label.append(int(dir))
                self.samples.append(os.path.join(self.image_dir, dir, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = default_loader(self.samples[idx])
        label = self.img_label[idx]
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        p_index1 = 0.9
        if np.random.random() < p_index1:
            index1 = 0
        else:
            index1 = np.random.randint(self.domain_num)
        index2 = np.random.randint(self.domain_num)
        while index2 == index1:
            index2 = np.random.randint(self.domain_num)
        img_3channel = img.split()
        img1 = Image.merge('RGB', (img_3channel[index_channel[index1][0]], img_3channel[index_channel[index1][1]],
                                   img_3channel[index_channel[index1][2]]))
        img2 = Image.merge('RGB', (img_3channel[index_channel[index2][0]], img_3channel[index_channel[index2][1]],
                                   img_3channel[index_channel[index2][2]]))
        if self.transform is not None:
            img1 = self.transform(img1)
        if self.transform is not None:
            img2 = self.transform(img2)
        label1 = self.class_num * index1 + label
        label2 = self.class_num * index2 + label
        # The below operation can produce data with more diversity
        if np.random.randint(2) == 0:
            return img1, img2, label1, label2
        else:
            return img2, img1, label2, label1



#############################################################################################################
# RandomErasing: Executing random erasing on input data
# Parameters
#         ----------
#         probability: The probability that the Random Erasing operation will be performed
#         sl: Minimum proportion of erased area against input image
#         sh: Maximum proportion of erased area against input image
#         r1: Minimum aspect ratio of erased area
#         mean: Erasing value
# -----------------------------------------------------------------------------------------------------------
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
