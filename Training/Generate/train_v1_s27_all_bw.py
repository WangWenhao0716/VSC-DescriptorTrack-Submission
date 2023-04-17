import os
from PIL import Image
import argparse

path = '../images/train_v1_s27_all/train_v1_s27_all/'
path_bw = '../images/train_v1_s27_all_bw/train_v1_s27_all_bw/'

os.makedirs(path_bw, exist_ok=True)

ls = sorted(os.listdir(path))

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * 10000
end = (num+1) * 10000


for i in range(begin, end):
    image = Image.open(path + ls[i])
    gray_image = image.convert("L")
    gray_image.save(path_bw + ls[i], quality = 100)
