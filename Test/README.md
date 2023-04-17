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

3. 
