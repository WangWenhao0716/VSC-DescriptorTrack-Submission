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
from models.nat_sup import nat_base

__all__ = ['NAT', 'NAT_base']

class NAT(nn.Module):
    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(NAT, self).__init__()
        self.pretrained = True
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        nat = nat_base(pretrained=False)
        
        nat.head = nn.Sequential()
        
        self.base = nn.Sequential(
            nat
        )#.cuda()
        
        self.linear = nn.Linear(1024, 512)
        
        self.classifier = build_metric('cos', 1024, self.num_classes, s=64, m=0.35).cuda()
        self.classifier_1 = build_metric('cos', 512, self.num_classes, s=64, m=0.6).cuda()
        
        self.projector_feat_bn = nn.Sequential(
                nn.Identity()
            ).cuda()

        self.projector_feat_bn_1 = nn.Sequential(
                self.linear,
                nn.Identity()
            ).cuda()
        

    def forward(self, x, y=None):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        
        bn_x = self.projector_feat_bn(x)
        prob = self.classifier(bn_x, y)
        
        bn_x_512 = self.projector_feat_bn_1(bn_x)
        prob_1 = self.classifier_1(bn_x_512, y)
        
        
        return bn_x_512, prob, prob_1


def NAT_base(**kwargs):
    return NAT(50, **kwargs)
