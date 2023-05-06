from __future__ import absolute_import

from .resnet import *
from .resneSt import *
from .resneXt import *
from .resnet_ibn import *
from .vit import *
#from .pvt_v2 import *
from .swin import swin_base
#from .uniformer import *
#from .nat import *
from .t2t import T2T_vit_14
from .cotnet import *
from .sknet import * 
from .iresnet import *
from .resnet_cbam import *
#from .hrnet import *
from .pyresnet import *
#from .cae import *

__factory = {
    'resnet50': resnet50,
    'resneSt50': resneSt50,
    'resneXt50': resneXt50,
    'cotnet50': cotnet50,
    'resnet_ibn50a': resnet_ibn50a,
    'iresnet50': iresnet50,
    'sknet50': sknet50,
    'resnet50_cbam': resnet50_cbam,
    #'hrnet_w30': hrnet_w30,
    'pyresnet50': pyresnet50,
    #cae_base': cae_base,
    'vit_base': vit_base,
    #'uni_base': uni_base,
    #'PVT_v2_b2': PVT_v2_b2,
    #'NAT_base': NAT_base,
    'T2T_vit_14': T2T_vit_14,
    'swin_base': swin_base,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
