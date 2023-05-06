from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from .gem import GeneralizedMeanPoolingP
from .metric import build_metric

__all__ = ['ResNeSt', 'resneSt18', 'resneSt34', 'resneSt50', 'resneSt101',
           'resneSt152']

class Waveblock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(0.3 * h)
            sx = random.randint(0, h-rh)
            mask = (x.new_ones(x.size()))*1.5
            mask[:, :, sx:sx+rh, :] = 1
            x = x * mask 
        return x

class ResNeSt(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(ResNeSt, self).__init__()
        self.pretrained = True
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        
        #resnet = ResNet.__factory[depth](pretrained=True)
        
        from resnest.torch import resnest50
        resneSt = resnest50(pretrained=True) 
        resneSt.layer4[0].avd_layer.stride = (1,1)
        resneSt.layer4[0].downsample[0].stride = (1,1)
        resneSt.layer4[0].downsample[0].kernel_size = (1,1)
        
        gap = GeneralizedMeanPoolingP() #nn.AdaptiveAvgPool2d(1)
        print("The init norm is ",gap)
        waveblock = Waveblock()
        
        self.base = nn.Sequential(
            resneSt.conv1, resneSt.bn1, resneSt.maxpool, resneSt.relu,
            resneSt.layer1,
            resneSt.layer2, waveblock,
            resneSt.layer3, waveblock,
            resneSt.layer4, gap
        ).cuda()
        
        self.linear = nn.Linear(2048, 512)
        
        if not self.cut_at_pooling:
            self.num_features = 2048
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resneSt.fc.in_features

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = build_metric('cos', 2048, self.num_classes, s=64, m=0.35).cuda()
                self.classifier_1 = build_metric('cos', 512, self.num_classes, s=64, m=0.35).cuda()
                
            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = 2048
                self.num_features_1 = 512
                feat_bn = nn.BatchNorm1d(self.num_features)
                feat_bn_1 = nn.BatchNorm1d(self.num_features_1)
                
            feat_bn.bias.requires_grad_(False)
            feat_bn_1.bias.requires_grad_(False)

            
        init.constant_(feat_bn.weight, 1)
        init.constant_(feat_bn.bias, 0)
        
        self.projector_feat_bn = nn.Sequential(
            feat_bn
        ).cuda()
        
        self.projector_feat_bn_1 = nn.Sequential(
            self.linear,
            feat_bn_1
        ).cuda()
        
    def forward(self, x, y=None):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        
        bn_x = self.projector_feat_bn(x)
        prob = self.classifier(bn_x, y)
        
        bn_x_512 = self.projector_feat_bn_1(bn_x)
        prob_1 = self.classifier_1(bn_x_512, y)
        
        
        return bn_x, prob, prob_1

def resneSt18(**kwargs):
    return ResNeSt(18, **kwargs)


def resneSt34(**kwargs):
    return ResNeSt(34, **kwargs)


def resneSt50(**kwargs):
    return ResNeSt(50, **kwargs)


def resneSt101(**kwargs):
    return ResNeSt(101, **kwargs)


def resneSt152(**kwargs):
    return ResNeSt(152, **kwargs)