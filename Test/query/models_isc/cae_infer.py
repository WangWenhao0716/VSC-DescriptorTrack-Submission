import os
import math
import torch
import torch.nn.functional as F
from timm.models import create_model
from .cae_model import cae_tiny_patch16_224, cae_small_patch16_224, cae_base_patch16_224, cae_large_patch16_224
from .cae_model import cae_tiny_patch16_192, cae_small_patch16_192, cae_base_patch16_192, cae_large_patch16_192

__models__ = {
    "cae_tiny_patch16_192": cae_tiny_patch16_192,
    "cae_small_patch16_192": cae_small_patch16_192,
    "cae_base_patch16_192": cae_base_patch16_192,
    "cae_large_patch16_192": cae_large_patch16_192,
    "cae_tiny_patch16_224": cae_tiny_patch16_224,
    "cae_small_patch16_224": cae_small_patch16_224,
    "cae_base_patch16_224": cae_base_patch16_224,
    "cae_large_patch16_224": cae_large_patch16_224,    
}

__drop_path_rates__ = {
    'cae_tiny_patch16_192': 0.0,
    'cae_small_patch16_192': 0.1,
    'cae_base_patch16_192': 0.1,
    'cae_large_patch16_192': 0.2,
    'cae_tiny_patch16_224': 0.0,
    'cae_small_patch16_224': 0.1,
    'cae_base_patch16_224': 0.1,
    'cae_large_patch16_224': 0.2
}

def get_valid_parameters(torch_params, suffix='pretrain'):
    new_params = {}
    if suffix == 'finetune':
        for key, value in torch_params.items():
            if 'fc_norm' in key:
                new_params[key.replace('fc_norm.', 'norm.')] = value
            else:
                new_params[key] = value
    elif suffix == 'pretrain':
        for key, value in torch_params.items():
            if 'encoder' in key and 'decoder' not in key:
                new_params[key.replace('encoder.', '')] = value
    else:
        new_params = torch_params
        
    return new_params

def resize_pos_embed(posemb, posemb_new, hight, width, hw_ratio):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    if posemb.dim() == 2:
        posemb = posemb.reshape(1, posemb.shape[0], posemb.shape[1])
    if posemb_new.dim() == 2:
        posemb_new = posemb_new.reshape(1, posemb_new.shape[0], posemb_new.shape[1])

    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old_h = int(math.sqrt(len(posemb_grid)*hw_ratio))
    gs_old_w = gs_old_h // hw_ratio
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)

    if posemb.dim() == 2:
        posemb = posemb.reshape(1, posemb.shape[1], posemb.shape[2])

    return posemb

def build_model(params_path, model_name='cae_base_patch16_224', sin_pos_emb=True, rel_pos_bias=False, height=14, width=14):
    suffix = ''
    if 'pretrain' in params_path:
        suffix = 'pretrain'
    if 'finetune' in params_path:
        suffix = 'finetune'
        
    torch_params = torch.load(params_path, map_location=torch.device('cpu'))
    
    if 'arch' in torch_params:
        model_name = torch_params['arch']
    ### get parameters ###
    if 'state_dict' in torch_params:
        torch_params = torch_params['state_dict']
    elif 'model_dict' in torch_params:
        torch_params = torch_params['model_dict']
    elif 'model' in torch_params:
        torch_params = torch_params['model']

    ### get valid parameters ###
    torch_params = get_valid_parameters(torch_params, suffix=suffix)
    torch_model = create_model(model_name, drop_path_rate=__drop_path_rates__[model_name], init_values=0.1, sin_pos_emb=sin_pos_emb, use_rel_pos_bias=rel_pos_bias)

    if torch_params['pos_embed'].shape != torch_model.state_dict()['pos_embed'].shape:
        torch_params['pos_embed'] = resize_pos_embed(torch_params['pos_embed'], torch_model.state_dict()['pos_embed'], height, width, hw_ratio=1)

    msg = torch_model.load_state_dict(torch_params, strict=True)
    print(msg)
    return torch_model

if __name__ == '__main__':
    model_name = 'cae_base_patch16_192'
    params_path = 'caev2_base_pretrain.pth'
    build_model(params_path, model_name=model_name, sin_pos_emb=True, rel_pos_bias=False, height=int(model_name.split('_')[-1])//16, width=int(model_name.split('_')[-1])//16)