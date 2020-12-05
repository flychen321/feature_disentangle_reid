# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import scipy.io
import yaml
import torch
from torchvision import datasets, transforms
from model import ft_net_dense
from model import load_network
from model import Auto_Encoder, decoder

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--which_epoch', default='best', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='market', type=str, help='./test_data')
parser.add_argument('--name', default='ide', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--net_loss_model', default=10, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=5, type=int, help='domain_num')

opt = parser.parse_args()
print('opt = %s' % opt)
print('opt.which_epoch = %s' % opt.which_epoch)
print('opt.test_dir = %s' % opt.test_dir)
print('opt.name = %s' % opt.name)
print('opt.batchsize = %s' % opt.batchsize)
###load config###
# load the training config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

name = opt.name
data_dir = os.path.join('data', opt.test_dir, 'pytorch')
print('data_dir = %s' % data_dir)

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_list = ['gallery', 'query']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in dataset_list}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in dataset_list}
class_names = image_datasets[dataset_list[1]].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 1024).zero_().cuda()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = img.cuda()
            outputs = model(input_img)
            ff = ff + outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.detach().cpu().float()
        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        if 'msmt' in opt.test_dir:
            camera = filename[9:11]
        else:
            camera = filename.split('c')[1][0]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


dataset_path = []
for i in range(len(dataset_list)):
    dataset_path.append(image_datasets[dataset_list[i]].imgs)

dataset_cam = []
dataset_label = []
for i in range(len(dataset_list)):
    cam, label = get_id(dataset_path[i])
    dataset_cam.append(cam)
    dataset_label.append(label)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
class_num = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
# class_num = 751
class_num *= opt.domain_num
encoder_a = ft_net_dense(class_num)
encoder_s = ft_net_dense(class_num)
decoder_a = decoder(2048)
model = Auto_Encoder(encoder_a, encoder_s, decoder_a)
if use_gpu:
    model.cuda()

if 'best' in opt.which_epoch or 'last' in opt.which_epoch:
    model = load_network(model, name, opt.which_epoch + '_' + str(opt.net_loss_model))
else:
    model = load_network(model, name, opt.which_epoch)
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
dataset_feature = []
with torch.no_grad():
    for i in range(len(dataset_list)):
        dataset_feature.append(extract_feature(model, dataloaders[dataset_list[i]]))


result = {'gallery_f': dataset_feature[0].numpy(), 'gallery_label': dataset_label[0], 'gallery_cam': dataset_cam[0],
          'query_f': dataset_feature[1].numpy(), 'query_label': dataset_label[1], 'query_cam': dataset_cam[1]}
scipy.io.savemat('pytorch_result.mat', result)
