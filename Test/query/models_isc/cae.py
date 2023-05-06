from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from .metric import build_metric

from .cae_infer import build_model

__all__ = ['CAETransformer', 'cae_base']


class CAETransformer(nn.Module):

    def __init__(self, weight, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(CAETransformer, self).__init__()
        self.pretrained = True
        self.weight = weight
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
       
        model_name = 'cae_base_patch16_192'
        params_path = 'pretrain_cae.pth'
        
        vit = build_model(params_path, model_name=model_name, sin_pos_emb=True, rel_pos_bias=False, \
            height=int(model_name.split('_')[-1])//16, width=int(model_name.split('_')[-1])//16)
        
        
        self.base = nn.Sequential(
            vit
        ).cuda()
        self.linear = nn.Linear(768, 512)
        
        self.classifier = build_metric('cos', 768, self.num_classes, s=64, m=0.35).cuda()
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

def cae_base(**kwargs):
    return CAETransformer('base', **kwargs)