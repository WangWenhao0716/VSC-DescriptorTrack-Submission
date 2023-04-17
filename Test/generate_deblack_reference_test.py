import os
import numpy as np
from random import choice
from PIL import Image
import shutil

path = '/raid/VSC/data/test/reference_one_second_imageio_v3_pad/'
path_new = '/raid/VSC/data/test/reference_one_second_imageio_v3_padcrop/'

os.makedirs(path_new, exist_ok=True)

all_detection = os.listdir(path)
all_detection = [i for i in all_detection if 'jpg.npy' in i]

for d in range(len(all_detection)):
    if d%1000==0:
        print(d)
    det = all_detection[d]
    detect = np.load(path + det)
    overylay = detect[detect[:,-1]==0]
    if len(overylay)>=1:
        overylay_select = overylay[0]
        old_img = Image.open(path + det[:-4])
        w, h = old_img.size
        max_length = max(w,h)
        enlarge = 640/max_length
        new_w = int(enlarge*w)
        new_h = int(enlarge*h)
        old_img = old_img.resize((new_w,new_h))
        new_img = old_img.crop(overylay_select[:-2])
        new_img.save(path_new +  det[:-4], quality=100)
