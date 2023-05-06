import pandas as pd
import numpy as np

import os
import shutil
import imageio
from PIL import Image
from augly.image.transforms import *
from random import choice

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import Counter

from isc.io import read_descriptors, write_hdf5_descriptors
import faiss

class ImageList(Dataset):
        def __init__(self, image_list, imsize=None, transform=None):
            Dataset.__init__(self)
            self.image_list = image_list
            self.transform = transform
            self.imsize = imsize

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, i):
            x = Image.open(self.image_list[i])
            x = x.convert("RGB")
            if self.imsize is not None:
                x = x.resize((self.imsize,self.imsize))
            if self.transform is not None:
                x = self.transform(x)
            return x
        
swintransformer = timm.create_model('swin_small_patch4_window7_224', pretrained=False)
swintransformer.head = nn.Linear(768,4)
swintransformer = swintransformer.eval()
ckpt = torch.load('rotate_detect.pth',map_location='cpu')
swintransformer.load_state_dict(ckpt["model"])
swintransformer.cuda()
t = []
t.append(transforms.Resize((224,224)))
t.append(transforms.ToTensor())
t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
tran = transforms.Compose(t)
image_dir = '/dev/shm/query_one_second_imageio_v3_detection_v3/'
namess = os.listdir(image_dir)
image_list = sorted([image_dir + '/' + name for name in namess])
im_dataset = ImageList(image_list, transform=tran, imsize=224)
dataloader = torch.utils.data.DataLoader(
        im_dataset, batch_size=256, shuffle=False,
        num_workers=4
    )
with torch.no_grad():
    all_desc = []
    for i, x in enumerate(dataloader):
        prob = nn.Softmax(dim=1)(swintransformer(x.cuda()))
        all_desc.append(torch.argmax(prob, dim=1).cpu())
all_desc = torch.hstack(tuple(all_desc)).cpu().numpy()

path_1 = '/dev/shm/query_one_second_imageio_v3_detection_v3_rotate_1/'
os.makedirs(path_1, exist_ok = True)
for q in range(len(image_list)):
    query = image_list[q]
    label = all_desc[q]
    dst = path_1 + image_list[q].split('//')[1]
    if label == 0:
        src = query
        shutil.copy(src, dst)
    elif label == 1:
        img = Image.open(query).resize((256,256)).rotate(180)
        img.save(dst, quality=100)
    else:
        img = Image.open(query).resize((256,256)).rotate(90)
        img.save(dst, quality=100)
image_dir = path_1
namess = os.listdir(image_dir)
image_list = sorted([image_dir + '/' + name for name in namess])
im_dataset = ImageList(image_list, transform=tran, imsize=224)
dataloader = torch.utils.data.DataLoader(
        im_dataset, batch_size=256, shuffle=False,
        num_workers=4
    )
with torch.no_grad():
    all_desc = []
    for i, x in enumerate(dataloader):
        prob = nn.Softmax(dim=1)(swintransformer(x.cuda()))
        all_desc.append(torch.argmax(prob, dim=1).cpu())
all_desc = torch.hstack(tuple(all_desc)).cpu().numpy()

path_2 = '/dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2/'
os.makedirs(path_2, exist_ok = True)
for q in range(len(image_list)):
    query = image_list[q]
    label = all_desc[q]
    dst = path_2 + image_list[q].split('//')[1]
    if label == 0:
        src = query
        shutil.copy(src, dst)
    elif label == 1:
        img = Image.open(query).resize((256,256)).rotate(180)
        img.save(dst, quality=100)
    else:
        img = Image.open(query).resize((256,256)).rotate(-90)
        img.save(dst, quality=100)