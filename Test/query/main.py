import pandas as pd
import numpy as np

import os
import shutil
import imageio
import pickle
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

ROOT_DIRECTORY = "/code_execution/"
DATA_DIRECTORY = "/data/"
QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY + "query/"
OUTPUT_FILE = ROOT_DIRECTORY + "subset_query_descriptors.npz"
QUERY_SUBSET_FILE = DATA_DIRECTORY + "query_subset.csv"


def tranditional_filter():
    os.system("conda run --no-capture-output -n condaenv python tranditional_filter.py")
    #os.system("./tranditional_filter")

def videos_to_images(query_subset_video_ids):
    path_save = '/dev/shm/query_one_second_imageio_v3/'
    os.makedirs(path_save, exist_ok = True)
    ls = [q + '.mp4' for q in query_subset_video_ids]
    for i in range(len(ls)):
        vid = imageio.get_reader(QRY_VIDEOS_DIRECTORY + ls[i],  'ffmpeg')
        num_frames = vid.count_frames()
        num_seconds = vid.get_meta_data()['duration']
        num_image_per_second = int(num_frames/num_seconds)
        for ii, im in enumerate(vid):
            if (ii%num_image_per_second==0 and (ii//num_image_per_second) <= num_seconds - 1):
                img = Image.fromarray(im)
                img.save(path_save + ls[i].split('.')[0] + '_' + str(ii//num_image_per_second) +'.jpg', quality=100)
class PPad:
    def __call__(self, x):
        w_factor = 0.1
        h_factor = 0.1
        color_1 = 255
        color_2 = 255
        color_3 = 255
        x = Pad(w_factor = w_factor, h_factor = h_factor, color = (color_1, color_2, color_3))(x)
        return x
    
def detectection():
    ls = sorted(os.listdir('/dev/shm/query_one_second_imageio_v3'))
    os.makedirs('/dev/shm/query_one_second_imageio_v3_detection_media_v3/', exist_ok = True)
    for i in range(len(ls)):
        name = ls[i]
        if int(name.split('.')[0].split('_')[1])%2==1:
            src = '/dev/shm/query_one_second_imageio_v3/' + name
            dst = '/dev/shm/query_one_second_imageio_v3_detection_media_v3/' + name
            img = Image.open(src)
            img = PPad()(img)
            img.save(dst, quality=100)
            
    os.makedirs("/dev/shm/query_one_second_imageio_v3_detection_v3", exist_ok = True)
    for i in range(len(ls)):
        name = ls[i]
        if int(name.split('.')[0].split('_')[1])%2==1:
            src = '/dev/shm/query_one_second_imageio_v3/' + name
            dst = '/dev/shm/query_one_second_imageio_v3_detection_v3/' + name
            shutil.copy(src, dst)
    os.system('conda run --no-capture-output -n condaenv \
    python detect.py --source /dev/shm/query_one_second_imageio_v3_detection_media_v3 \
    --weights best_20230101.pt --conf 0.1 > /dev/null')
    
def generation():
    path = '/dev/shm/query_one_second_imageio_v3_detection_media_v3/'
    path_new = '/dev/shm/query_one_second_imageio_v3_detection_v3/'
    path_ori = '/dev/shm/query_one_second_imageio_v3/'
    all_detection = os.listdir(path)
    all_detection = [i for i in all_detection if 'jpg.npy' in i]
    for d in range(len(all_detection)):
        det = all_detection[d]
        detect = np.load(path + det)
        overylay = detect[detect[:,-1]==0]
        if len(overylay)>=1:
            overylay_select = choice(overylay)
            old_img = Image.open(path + det[:-4])
            w, h = old_img.size
            max_length = max(w,h)
            enlarge = 640/max_length
            new_w = int(enlarge*w)
            new_h = int(enlarge*h)
            old_img = old_img.resize((new_w,new_h))
            new_img = old_img.crop(overylay_select[:-2])
            new_img.save(path_new +  det[:-4], quality=100)
    ls = sorted(os.listdir('/dev/shm/query_one_second_imageio_v3'))
    for i in range(len(ls)):
        name = ls[i]
        if int(name.split('.')[0].split('_')[1])%2==0:
            src = path_ori + name
            dst = path_new + name
            shutil.copy(src, dst)
    assert len(os.listdir(path_new)) == len(os.listdir(path_ori))
    
def fix_rotate():
    os.system("conda run --no-capture-output -n condaenv python fix_rotate.py")
            
def extract_feature_vit():
    os.makedirs('./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2 \
      --o ./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/query_v1_detection_rotate.hdf5 \
      --model vit_base  --GeM_p 3 --bw \
      --checkpoint train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar --imsize 224')

def extract_feature_swin():
    os.makedirs('./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2 \
      --o ./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/query_v1_detection_rotate.hdf5 \
      --model swin_base  --GeM_p 3 --bw \
      --checkpoint train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar --imsize 224')
    
def extract_feature_t2t():
    os.makedirs('./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2 \
      --o ./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/query_v1_detection_rotate.hdf5 \
      --model t2t  --GeM_p 3 --bw \
      --checkpoint train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar --imsize 224')
    
def extract_feature_50():
    os.makedirs('./feature/train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2 \
      --o ./feature/train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_v1_detection_rotate.hdf5 \
      --model 50  --GeM_p 3 --bw \
      --checkpoint train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar --imsize 256')

def extract_feature_50SK():
    os.makedirs('./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2 \
      --o ./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_v1_detection_rotate.hdf5 \
      --model 50SK  --GeM_p 3 --bw \
      --checkpoint train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar --imsize 256')

def extract_feature_50X():
    os.makedirs('./feature/train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_imageio_v3_detection_v3_rotate_2 \
      --o ./feature/train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_v1_detection_rotate.hdf5 \
      --model 50X  --GeM_p 3 --bw \
      --checkpoint train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar --imsize 256')    
    
def ensemble():
    num = len(os.listdir("./feature/train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN"))
    
    name, resnet =read_descriptors(
        ['./feature/train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_%d_v1_detection_rotate.hdf5'%i for i in range(num)]
    )
    
    name, resneXt =read_descriptors(
        ['./feature/train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_%d_v1_detection_rotate.hdf5'%i for i in range(num)]
    )
    
    name, sknet =read_descriptors(
        ['./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_%d_v1_detection_rotate.hdf5'%i for i in range(num)]
    )
    
    name, vit =read_descriptors(
        ['./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/query_%d_v1_detection_rotate.hdf5'%i for i in range(num)]
    )
    
    name, swin =read_descriptors(
        ['./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/query_%d_v1_detection_rotate.hdf5'%i for i in range(num)]
    )
    
    name, t2t =read_descriptors(
        ['./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/query_%d_v1_detection_rotate.hdf5'%i for i in range(num)]
    )
    
    select_features = []
    for i in range(len(name)):
        f_2 = resnet[i].reshape((512,1))
        f_3 = resneXt[i].reshape((512,1))
        f_7 = sknet[i].reshape((512,1))
        f_11 = vit[i].reshape((512,1))
        f_12 = t2t[i].reshape((512,1))
        f_13 = swin[i].reshape((512,1))
        con = np.concatenate((f_2, f_3, f_7, f_11, f_12, f_13),axis=1)
        new = np.mean(con, axis=1).reshape(1,512)
        select_features.append(new)
    feature_f = np.concatenate(select_features,axis=0)
    faiss.normalize_L2(feature_f)
    write_hdf5_descriptors(feature_f, name, 'query_subset_feature.hdf5')

def score_normalization():
    os.system("CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n condaenv python score_normalization_test.py \
    --query_descs query_subset_feature.hdf5 \
    --train_descs ensemble_6_t_512_crop_rotate_bw_gt_ng_1_cls_FIN.hdf5 \
    --factor 2 --nn 10 \
    --out_path ./ \
    --o ./predictions_train_subset.csv \
    --reduction avg --max_results 2000_000")
    
def main():
    import time
    # Loading subset of query images
    begin = time.time()
    #query_subset = pd.read_csv(QUERY_SUBSET_FILE)
    #query_subset_video_ids = query_subset.video_id.values.astype("U")

    print("Install packages")
    os.system("conda run --no-capture-output -n condaenv python -m pip install imageio_ffmpeg-0.4.8-py3-none-manylinux2010_x86_64.whl --force-reinstall")
    os.system("conda run --no-capture-output -n condaenv python -m pip install timm-0.4.12-py3-none-any.whl")
    #os.system("conda uninstall opencv -n condaenv -y")
    #os.system("conda run --no-capture-output -n condaenv python -m pip install opencv_python-4.5.4.60-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps --force-reinstall")
    
    print("Perform traditional filters!")
    tranditional_filter()
    with open('tranditional_filter.pkl', 'rb') as f:
        query_subset_video_ids = pickle.load(f)
    query_subset_video_ids = sorted(list(set(query_subset_video_ids)))
    print(len(query_subset_video_ids))

    print("Transfer videos to images every one second using imageio")
    videos_to_images(query_subset_video_ids)
    
    print("Using detection models to generate new test sets")
    detectection()
    generation()
    
    print("Fix rotating")
    fix_rotate()

    print("Extract features")
    os.makedirs('feature/', exist_ok = True)
    extract_feature_vit()
    extract_feature_swin()
    extract_feature_t2t()
    extract_feature_50()
    extract_feature_50X()
    extract_feature_50SK()

    print("Ensemble")
    ensemble()
    
    print("Normalize and Transfer hdf5 to npz")
    score_normalization()
    print("Total time = ", time.time()-begin)

if __name__ == "__main__":
    main()
