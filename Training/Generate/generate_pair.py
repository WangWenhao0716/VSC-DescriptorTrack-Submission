import pandas as pd
import os
import cv2
from PIL import Image

def select_image(time, path):
    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    success, frame = video.read()
    return frame


df = pd.read_csv('/raid/VSC/data/train/train_matching_ground_truth.csv')
path_q = '/raid/VSC/data/train/query/'
path_r = '/raid/VSC/data/train/reference/'

os.system("mkdir -p /raid/VSC/data/pull_gt_all/gt_query")
os.system("mkdir -p /raid/VSC/data/pull_gt_all/gt_ref")

count = 0
for num in range(len(df)):
    print(num)
    rate = (df.iloc[num]['ref_end'] - df.iloc[num]['ref_start'])\
    /(df.iloc[num]['query_end'] - df.iloc[num]['query_start'])
    
    uu = 1
    while (df.iloc[num]['query_start'] + 1 * uu < df.iloc[num]['query_end']) and \
          (df.iloc[num]['ref_start'] + rate * uu < df.iloc[num]['ref_end']):
        img_2_q = select_image(df.iloc[num]['query_start'] + 1 * uu, path_q + df.iloc[num]['query_id'] + '.mp4')
        img_2_r = select_image(df.iloc[num]['ref_start'] + rate * uu, path_r + df.iloc[num]['ref_id'] + '.mp4')
        os.makedirs('/raid/VSC/data/pull_gt_all/gt_query/'+str(count), exist_ok = True)
        os.makedirs('/raid/VSC/data/pull_gt_all/gt_ref/'+str(count), exist_ok = True)
        name_q = '/raid/VSC/data/pull_gt_all/gt_query/' + str(count) + '/' + df.iloc[num]['query_id'] + '.jpg'
        name_r = '/raid/VSC/data/pull_gt_all/gt_ref/' + str(count) + '/' + df.iloc[num]['ref_id'] + '.jpg'
        try:
            cv2.imwrite(name_q, img_2_q, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(name_r, img_2_r, [cv2.IMWRITE_JPEG_QUALITY, 100])
            count = count + 1
            uu = uu + 1
        except:
            uu = uu + 1
            print('fail!')
