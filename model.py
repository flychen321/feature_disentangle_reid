import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
from collections import OrderedDict


######################################################################
# Load parameters of model
# ---------------------------
def load_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Load model
# ---------------------------
def load_whole_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


######################################################################
# Save parameters of model
# ---------------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)

######################################################################
# Save model
# ---------------------------
def save_whole_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network, save_path)


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.detach(), std=0.001)
        init.constant_(m.bias.detach(), 0.0)

######################################################################
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# --------------------------------------------------------------------
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f

######################################################################
# Define the DenseNet121-based Model
# --------------------------------------------------------------------
class ft_net_dense(nn.Module):
    def __init__(self, class_num=751):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Fc_ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        mid_coder = x
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        f = x
        f_norm = \
            f.norm(p=2, dim=1, keepdim=True) + 1e-8
        feature_coder = f.div(f_norm)
        x = self.classifier(x)
        return x[0], x[1], mid_coder, feature_coder


######################################################################
# Auto_Encoder is used to disentangle different semantic feature
# self.embedding_net_c have the same network structure with self.embedding_net_s
# self.embedding_net_c is used for extracting content-related information
# self.embedding_net_s is used for extracting sketch-related information
# self.decoder is used for reconstructing the original images
# --------------------------------------------------------------------
class Auto_Encoder(nn.Module):
    def __init__(self, embedding_net_c, embedding_net_s, decoder):
        super(Auto_Encoder, self).__init__()
        self.embedding_net_c = embedding_net_c
        self.embedding_net_s = embedding_net_s
        self.decoder = decoder

    def forward(self, x_1, x_2=None):
        output_c1, feature_c1, mid_coder_c1, feature_coder_c1 = self.embedding_net_c(x_1)
        output_s1, feature_s1, mid_coder_s1, feature_coder_s1 = self.embedding_net_s(x_1)
        if x_2 is None:
            return torch.cat((feature_c1, feature_s1), 1)
        output_c2, feature_c2, mid_coder_c2, feature_coder_c2 = self.embedding_net_c(x_2)
        output_s2, feature_s2, mid_coder_s2, feature_coder_s2 = self.embedding_net_s(x_2)
        mid = torch.cat((mid_coder_c1, mid_coder_s1), 1)
        rec_img_cs11 = self.decoder(mid)
        mid = torch.cat((mid_coder_c1, mid_coder_s2), 1)
        rec_img_cs12 = self.decoder(mid)
        mid = torch.cat((mid_coder_c2, mid_coder_s2), 1)
        rec_img_cs21 = self.decoder(mid)
        mid = torch.cat((mid_coder_c2, mid_coder_s1), 1)
        rec_img_cs22 = self.decoder(mid)
        return output_c1, output_c2, output_s1, output_s2, \
               rec_img_cs11, rec_img_cs12, rec_img_cs21, rec_img_cs22,\
               feature_coder_c1, feature_coder_c2, feature_coder_s1, feature_coder_s2

#############################################################################################################
# decoder is used for reconstructing the original images, ant it consists 5 transposed convolutional layers
# The input size is 2048*8*4
# The output size is 3*256*128
# -----------------------------------------------------------------------------------------------------------
class decoder(nn.Module):
    def __init__(self, in_channels=2048):
        super(decoder, self).__init__()
        layer0 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(128)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer2 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer3 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer4 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(3)),
        ]))
        layer0.apply(weights_init_kaiming)
        layer1.apply(weights_init_kaiming)
        layer2.apply(weights_init_kaiming)
        layer3.apply(weights_init_kaiming)
        layer4.apply(weights_init_kaiming)

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


