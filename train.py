# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import os
import numpy as np
import yaml
from model import ft_net_dense
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from model import save_network, Auto_Encoder, decoder
from losses import ContrastiveLoss_CS, SoftLabelLoss
from datasets import Channel_Dataset, RandomErasing
version = torch.__version__

######################################################################
# Options
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='ide', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='market', type=str, help='training dir path')
parser.add_argument('--batchsize', default=24, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=100, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=6, type=int, help='domain_num, in [2,6]')
parser.add_argument('--class_base', default=751, type=int, help='class_base, in [751, 702, 767]')

opt = parser.parse_args()
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
print('domain_num = %s' % opt.domain_num)
if opt.domain_num > 6 or opt.domain_num < 2:
    print('domain_num = %s' % opt.domain_num)
    exit()
data_dir = os.path.join('data', opt.data_dir, 'pytorch')
name = opt.name

opt.class_base = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
######################################################################
# Load Data
# --------------------------------------------------------------------
#
transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
}


image_datasets = {}
image_datasets['train'] = Channel_Dataset(os.path.join(data_dir, 'train_all_new'),
                                  data_transforms['train'], domain_num=opt.domain_num)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8) for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
use_gpu = torch.cuda.is_available()

#######################################################################################################################
# convert 'one-hot' label to soft-label
# Parameters
#         ----------
#         labels : determine which bit correspondings to '1' in 'one-hot' label
#         w_main : determine the value of main ingredient bit
#         w_content : It is the total probablity of content-related information, and it will be aliquoted by all bits
#                     which corresponding to the categories with the same channnel order
#         w_sketch: It is the total probablity of sketch-related information, and it will be aliquoted by all bits
#                   which corresponding to the categories with the same sketch
#         domain_num: determine how many channel orders the soft-label includes
# ---------------------------------------------------------------------------------------------------------------------
def get_soft_label_6domain(labels, w_main=0.7, w_content=0.289999, w_sketch=0.01, domain_num=2):
    class_base = opt.class_base
    # w_reg is used to prevent data overflow, it a value close to zero
    w_reg = (1 - w_main - w_sketch - w_content) / (domain_num * class_base)
    w_sketch /= domain_num
    if w_reg < 0:
        print('w_main=%s   w_content=%s   w_sketch=%s' % (w_main, w_content, w_sketch))
        exit()
    w_content /= class_base
    soft_label = np.zeros((len(labels), int(domain_num * class_base)))
    soft_label.fill(w_reg)
    for i in np.arange(len(labels)):
        base_label = labels[i] % class_base
        domain_order = labels[i] // class_base
        for j in np.arange(class_base):
            soft_label[i][domain_order * class_base + int(j)] = w_content + w_reg
        for k in np.arange(domain_num):
            if k == domain_order:
                soft_label[i][k * class_base + base_label] = w_main + w_content + w_reg
            else:
                soft_label[i][k * class_base + base_label] = w_sketch + w_reg
    return torch.Tensor(soft_label)


######################################################################
# Training the model
# --------------------------------------------------------------------
def train(model, criterion_identify, criterion_reconstruct, criterion_contrastive, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    cnt = 0

    # Cross-dataset fine-tune
    # best for Market, evaluation for Duke, CUHK03 and MSMT17
    # r_id = 0.3
    # r_rec = 0.3
    # r_con = 1.0
    # r_s = 0.5
    # r_c = 0.2
    # w_main_c = 0.7
    # w_sketch_c = 0
    # w_main_s = 0.9
    # w_content_s = 0


    # Cross-dataset fine-tune
    # best for Duke, evaluation for Market
    r_id = 0.3
    r_rec = 0.2
    r_con = 1.0
    r_s = 0.6
    r_c = 0.2
    w_main_c = 0.7
    w_sketch_c = 0
    w_main_s = 0.95
    w_content_s = 0


    # r_id = 0  # for ablation, disable Id loss
    # r_c = 0  # for ablation, disable contrastive loss
    # r_s = 0  # for ablation, disable contrastive loss
    w_content_c = 1 - w_main_c - 1e-5
    w_sketch_s = 1 - w_main_s - 1e-5
    print('r_id = %.3f   r_rec = %.3f   r_con = %.3f   r_s = %.3f   r_c = %.3f' % (
        r_id, r_rec, r_con, r_s, r_c))
    print('w_main_c = %.3f   w_content_c = %.3f   w_sketch_c = %.3f' % (w_main_c, w_content_c, w_sketch_c))
    print('w_main_s = %.3f   w_content_s = %.3f   w_sketch_s = %.3f' % (w_main_s, w_content_s, w_sketch_s))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            scheduler.step()
            model.train(True)  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs1, inputs2, id_labels1, id_labels2 = data
                # get the soft-label for feature disentangling
                id_labels_content_1 = get_soft_label_6domain(id_labels1, w_main=w_main_c, w_content=w_content_c,
                                                             w_sketch=w_sketch_c, domain_num=opt.domain_num)
                id_labels_content_2 = get_soft_label_6domain(id_labels2, w_main=w_main_c, w_content=w_content_c,
                                                             w_sketch=w_sketch_c, domain_num=opt.domain_num)
                id_labels_sketch_1 = get_soft_label_6domain(id_labels1, w_main=w_main_s, w_content=w_content_s,
                                                            w_sketch=w_sketch_s, domain_num=opt.domain_num)
                id_labels_sketch_2 = get_soft_label_6domain(id_labels2, w_main=w_main_s, w_content=w_content_s,
                                                            w_sketch=w_sketch_s, domain_num=opt.domain_num)

                now_batch_size, c, h, w = inputs1.shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if use_gpu:
                    inputs1 = inputs1.cuda()
                    inputs2 = inputs2.cuda()
                    id_labels_content_1 = id_labels_content_1.cuda()
                    id_labels_content_2 = id_labels_content_2.cuda()
                    id_labels_sketch_1 = id_labels_sketch_1.cuda()
                    id_labels_sketch_2 = id_labels_sketch_2.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output_content_1, output_content_2, output_sketch_1, output_sketch_2, \
                rec_img_cs11, rec_img_cs12, rec_img_cs21, rec_img_cs22, \
                feature_coder_c1, feature_coder_c2, feature_coder_s1, feature_coder_s2 = model(inputs1, inputs2)

                _, id_preds_content_1 = torch.max(output_content_1.detach(), 1)
                _, id_preds_content_2 = torch.max(output_content_2.detach(), 1)
                _, id_preds_sketch_1 = torch.max(output_sketch_1.detach(), 1)
                _, id_preds_sketch_2 = torch.max(output_sketch_2.detach(), 1)
                loss_id = 0
                loss_id += criterion_identify(output_content_1, id_labels_content_1)
                loss_id += criterion_identify(output_content_2, id_labels_content_2)
                loss_id += criterion_identify(output_sketch_1, id_labels_sketch_1)
                loss_id += criterion_identify(output_sketch_2, id_labels_sketch_2)
                loss_rec = 0
                loss_rec += criterion_reconstruct(rec_img_cs11, inputs1)
                loss_rec += criterion_reconstruct(rec_img_cs12, inputs1)
                loss_rec += criterion_reconstruct(rec_img_cs21, inputs2)
                loss_rec += criterion_reconstruct(rec_img_cs22, inputs2)
                loss_c, loss_s = criterion_contrastive(feature_coder_c1, feature_coder_c2, feature_coder_s1,
                                                       feature_coder_s2)
                loss_con = r_s * loss_s + r_c * loss_c
                # calculate the total loss
                loss = r_id * loss_id + r_rec * loss_rec + loss_con

                if cnt % 200 == 0:
                    print('cnt = %5d   loss   = %.4f  loss_id = %.4f  loss_rec = %.4f  loss_con = %.4f' % (
                        cnt, loss.cpu().detach().numpy(), loss_id.cpu().detach().numpy(),
                        loss_rec.cpu().detach().numpy(), loss_con.cpu().detach().numpy()))
                    print('loss_c = %.4f  loss_s  = %.4f' % (
                        loss_c.cpu().detach().numpy(), loss_s.cpu().detach().numpy()))
                cnt += 1

                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                running_corrects += float(torch.sum(id_preds_content_1 == id_labels_content_1.argmax(1).detach()))
                running_corrects += float(torch.sum(id_preds_content_2 == id_labels_content_2.argmax(1).detach()))
                running_corrects += float(torch.sum(id_preds_sketch_1 == id_labels_sketch_1.argmax(1).detach()))
                running_corrects += float(torch.sum(id_preds_sketch_2 == id_labels_sketch_2.argmax(1).detach()))

            datasize = dataset_sizes[phase] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / (datasize * 4)

            print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, name, 'best' + '_' + str(opt.net_loss_model))

            save_network(model, name, epoch)

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    save_network(model, name, 'last' + '_' + str(opt.net_loss_model))


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 8 hours on GPU when batchsize = 24.
#---------------------------------------------------------------------
dir_name = os.path.join('./model', name)
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

class_num = image_datasets['train'].class_num
class_num *= opt.domain_num
print('class_num = %d' % class_num)
encoder_c = ft_net_dense(class_num)
encoder_s = ft_net_dense(class_num)
decoder = decoder(2048)
model = Auto_Encoder(encoder_c, encoder_s, decoder)
if use_gpu:
    model.cuda()

# print('model structure')
# print(model)

# Initialize loss functions
criterion_identify = SoftLabelLoss()
criterion_reconstruct = nn.MSELoss()
margin = 2.0
print('margin = %s' % margin)
criterion_contrastive = ContrastiveLoss_CS(margin)

# Set different learning rates for newly added parameters and based layers of DenseNet-121
encoder_c_id = list(map(id, model.embedding_net_c.classifier.parameters())) \
               + list(map(id, model.embedding_net_c.model.fc.parameters()))
encoder_s_id = list(map(id, model.embedding_net_s.classifier.parameters())) \
               + list(map(id, model.embedding_net_s.model.fc.parameters()))
decoder_id = list(map(id, model.decoder.parameters()))
new_id = encoder_c_id + encoder_s_id + decoder_id
classifier_params = filter(lambda p: id(p) in new_id, model.parameters())
base_params = filter(lambda p: id(p) not in new_id, model.parameters())

optimizer_ft = torch.optim.SGD([
    {'params': classifier_params, 'lr': 1 * opt.lr},
    {'params': base_params, 'lr': 0.1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# If you want the better performance, you can increase the training epochs slightly.
epoch = 70
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[30, 60, 65], gamma=0.1)
# train the model
train(model, criterion_identify, criterion_reconstruct, criterion_contrastive, optimizer_ft, exp_lr_scheduler,
      num_epochs=epoch)
