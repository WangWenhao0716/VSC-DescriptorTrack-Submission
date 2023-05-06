from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from .metric import build_metric

from timm.models import create_model
import models_isc.vit_models

__all__ = ['VisionTransformer', 'vit_tiny', 'vit_small', 'vit_base']


class VisionTransformer(nn.Module):

    def __init__(self, weight, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(VisionTransformer, self).__init__()
        self.pretrained = True
        self.weight = weight
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        if weight == 'tiny':
            vit = create_model(
                'deit_tiny_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif weight == 'small':
            vit = create_model(
                'deit_small_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif weight == 'base':
            vit = create_model(
                'deit_base_patch16_224',
                pretrained=False,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        else:
            print("Not implement!!!")
            exit()
        
        vit.head = nn.Sequential()
        
        self.base = nn.Sequential(
            vit
        ).cuda()
        
        self.linear = nn.Linear(768, 512)
        
        self.classifier = build_metric('cos', vit.embed_dim, self.num_classes, s=64, m=0.35).cuda()
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

def vit_tiny(**kwargs):
    return VisionTransformer('tiny', **kwargs)

def vit_small(**kwargs):
    return VisionTransformer('small', **kwargs)

def vit_base(**kwargs):
    return VisionTransformer('base', **kwargs)
