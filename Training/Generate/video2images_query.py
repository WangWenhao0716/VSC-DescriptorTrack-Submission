import os
import pylab
import imageio
import skimage
import numpy as np
from PIL import Image
import argparse

path = '/raid/VSC/data/train/query/'
path_save = '/raid/VSC/data/train/query_one_second_imageio_v3/'

ls = sorted(os.listdir(path))
os.makedirs(path_save, exist_ok = True)

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * (len(ls)//100 + 1)
end = (num+1) * (len(ls)//100 + 1)

for i in range(begin, end):
    print("Processing... %d"%i)
    vid = imageio.get_reader(path + ls[i],  'ffmpeg')
    num_frames = vid.count_frames()
    num_seconds = vid.get_meta_data()['duration']
    num_image_per_second = int(num_frames/num_seconds)
    for ii, im in enumerate(vid):
        if (ii%num_image_per_second==0 and (ii//num_image_per_second) <= num_seconds - 1):
            img = Image.fromarray(im)
            img.save(path_save + ls[i].split('.')[0] + '_' + str(ii//num_image_per_second) +'.jpg', quality=100)
