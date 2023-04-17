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



__all__ = ['Swin', 'SwinTransformer']


class Swin(nn.Module):
    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_classes_small=0,
                 dev = None):
        super(Swin, self).__init__()
        self.pretrained = True
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        self.num_classes_small = num_classes_small
        # Construct base (pretrained) resnet
        #resnet = ResNet.__factory[depth](pretrained=True)
        
        
        print("Loading the supervised swin pre-trained model: ")
        import timm
        swintransformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        swintransformer.head = nn.Sequential()
        
        
        self.base = nn.Sequential(
            swintransformer
        #    pro
        )#.cuda()
        
        self.linear = nn.Linear(1024, 512)
        
        self.classifier = build_metric('cos', 1024, self.num_classes, s=64, m=0.35).cuda()
        self.classifier_1 = build_metric('cos', 512, self.num_classes, s=64, m=0.6).cuda()
        
        self.classifier_small = build_metric('cos', 1024, self.num_classes_small, s=64, m=0.35).cuda()
        self.classifier_small_1 = build_metric('cos', 512, self.num_classes_small, s=64, m=0.6).cuda()
        
        self.projector_feat_bn = nn.Sequential(
                nn.Identity()
            ).cuda()

        self.projector_feat_bn_1 = nn.Sequential(
                self.linear,
                nn.Identity()
            ).cuda()
        

    def forward(self, x, y=None, y_small=None):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        
        bn_x = self.projector_feat_bn(x)
        prob = self.classifier(bn_x, y)
        prob_small = self.classifier_small(bn_x, y_small)
        
        bn_x_512 = self.projector_feat_bn_1(bn_x)
        prob_1 = self.classifier_1(bn_x_512, y)
        prob_small_1 = self.classifier_small_1(bn_x_512, y_small)
        
        
        return bn_x_512, prob, prob_1, prob_small, prob_small_1


def swin_base_double(**kwargs):
    return Swin(50, **kwargs)
