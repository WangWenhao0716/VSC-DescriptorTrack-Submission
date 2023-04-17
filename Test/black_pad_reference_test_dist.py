import os
import shutil
from PIL import Image
from augly.image.transforms import *
import torchvision.transforms as transforms
import random
import augly.image as imaugs
from train_v1_s3 import *
import numpy as np
import csv
import argparse


ls = sorted(os.listdir('/raid/VSC/data/test/reference_one_second_imageio_v3'))

class PPad:
    
    def __call__(self, x):
        w_factor = 0.1
        h_factor = 0.1
        color_1 = 0
        color_2 = 0
        color_3 = 0
        x = Pad(w_factor = w_factor, h_factor = h_factor, color = (color_1, color_2, color_3))(x)
        return x

    
    
parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * (len(ls)//200 + 1)
end = (num+1) * (len(ls)//200 + 1)

os.makedirs('/raid/VSC/data/test/reference_one_second_imageio_v3_pad', exist_ok = True)
for i in range(begin, end):
    if i%1000==0:
        print(i)
    name = ls[i]
    src = '/raid/VSC/data/test/reference_one_second_imageio_v3/' + name
    dst = '/raid/VSC/data/test/reference_one_second_imageio_v3_pad/' + name
    img = Image.open(src)
    img = PPad()(img)
    img.save(dst, quality=100)
