import os, shutil
import argparse

ls = sorted(os.listdir('/raid/VSC/data/train/reference_one_second_imageio_v3/'))

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * (len(ls)//100 + 1)
end = (num+1) * (len(ls)//100 + 1)


for nu in range(100001 + begin, 100001 + end):
    if nu%10==0:
        print(nu)
    name = 'R' + str(nu)
    select = [i for i in ls if name in i]
    train_image = select[len(select)//2]
    src = '/raid/VSC/data/train/reference_one_second_imageio_v3/' + train_image
    dst = '/raid/VSC/data/training_images_ref1/' + train_image
    shutil.copy(src, dst)
    
