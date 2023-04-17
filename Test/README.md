# Test
You can directly download and unzip our [submitted file](https://drive.google.com/file/d/1--OST0kIrNsLBa7MCArztV9u46pKjcip/view?usp=share_link) (including all the trained models, and all other necessary files), then by running
```
conda run --no-capture-output -n condaenv python main.py
```
you will get the query features. 

However, because the file only includes extracted reference features and normalization features, we show how to get these two kinds of features here.

Assuming we have downloaded the training and test reference datasets, and stored as follows:

```
/raid/VSC/data/train/reference/
/raid/VSC/data/test/reference/
```
Also, we have $6$ models here:

1. ```train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

2. ```train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

3. ```train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

4. ```train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```

5. ```train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```

6. ```train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```


Then: 

1. We first transform ```/raid/VSC/data/test/reference/``` into images using ```imageio``` by:

```
bash video2images_ref.sh
```
Note we have transformed the ```/raid/VSC/data/train/reference/``` into images in the training section.

2. Get the training reference features (normalization features) by:

```
bash swin_train_ref.sh
bash vit_train_ref.sh
bash t2t_train_ref.sh
bash 50_train_ref.sh
bash 50X_train_ref.sh
bash 50SK_train_ref.sh
```
Then:
```
python agg_train_ref.py
```
You will get ```ensemble_6_t_512_crop_rotate_bw_gt_ng_1_cls_FIN.hdf5``` in the submission file.

3. Get the test reference features:

We first delete the black images:

(1) Generate pad images:
```
bash black_pad_reference_test_dist.sh
```
(2) Use yolov5 to detect:

Download ```deblack_20230107.pt``` from [Google Drive](https://drive.google.com/file/d/1Nn6xXh1I9Fp0hUpHjGEqeikCa8Wkzgif/view?usp=share_link).
```
CUDA_VISIBLE_DEVICES=0 python detect.py \
--source /raid/VSC/data/test/reference_one_second_imageio_v3_pad/ \
--weights deblack_20230107.pt --conf 0.1 
```
(3) Get final images:
```
python generate_deblack_reference_test.py
```

Then:

```
bash swin_test_ref.sh
bash vit_test_ref.sh
bash t2t_test_ref.sh
bash 50_test_ref.sh
bash 50X_test_ref.sh
bash 50SK_test_ref.sh
```
Then:
```
python agg_test_ref.py
```
You will get ```reference_descriptors.npz``` in the submission file.

