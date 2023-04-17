from __future__ import absolute_import

from .resnet import *
from .resnet_double import *
from .resneXt import *
from .resneXt_double import *
from .vit import *
from .vit_double import vit_base_double
from .swin import swin_base
from .swin_double import swin_base_double
from .cotnet import *
from .cotnet_pro import *
from .sknet import * 
from .sknet_double import * 
from .t2t import T2T_vit_14
from .t2t_double import T2T_vit_14_double
__factory = {
    'resnet50': resnet50,
    'resnet50_double': resnet50_double,
    'resneXt50': resneXt50,
    'resneXt50_double': resneXt50_double,
    'cotnet50': cotnet50,
    'cotnet50_pro': cotnet50_pro,
    'sknet50': sknet50,
    'sknet50_double': sknet50_double,
    'vit_base': vit_base,
    'T2T_vit_14': T2T_vit_14,
    'T2T_vit_14_double': T2T_vit_14_double,
    'vit_base_double': vit_base_double,
    'swin_base': swin_base,
    'swin_base_double': swin_base_double,
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
