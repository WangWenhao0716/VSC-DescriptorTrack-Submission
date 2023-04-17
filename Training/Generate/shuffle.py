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
    
 
ls_q = sorted(os.listdir('/raid/VSC/data/pull_gt_all_shuffle/gt_query/'))
dic_q = dict()
for i in range(len(ls_q)):
    dic_q[ls_q[i]] = os.listdir('/raid/VSC/data/pull_gt_all_shuffle/gt_query/' + ls_q[i])[0]

with open('pull_gt_all_shuffle_q.pickle', 'wb') as f:
    pickle.dump(dic_q, f)

ls_r = sorted(os.listdir('/raid/VSC/data/pull_gt_all_shuffle/gt_ref/'))
dic_r = dict()
for i in range(len(ls_r)):
    dic_r[ls_r[i]] = os.listdir('/raid/VSC/data/pull_gt_all_shuffle/gt_ref/' + ls_r[i])[0]

with open('pull_gt_all_shuffle_r.pickle', 'wb') as f:
    pickle.dump(dic_r, f)
    
