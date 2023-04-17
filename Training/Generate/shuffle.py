import os
import random
import shutil

ls = sorted(os.listdir('/raid/VSC/data/pull_gt_all/gt_query/'))
random.shuffle(ls)

q = sorted(os.listdir('/raid/VSC/data/pull_gt_all/gt_query/'))
r = sorted(os.listdir('/raid/VSC/data/pull_gt_all/gt_ref/'))


for i in range(len(q)):
    src = '/raid/VSC/data/pull_gt_all/gt_query/' + q[i]
    dst = '/raid/VSC/data/pull_gt_all_shuffle/gt_query/' + ls[i]
    shutil.copytree(src, dst)
    
for i in range(len(r)):
    src = '/raid/VSC/data/pull_gt_all/gt_ref/' + r[i]
    dst = '/raid/VSC/data/pull_gt_all_shuffle/gt_ref/' + ls[i]
    shutil.copytree(src, dst)
    
 
