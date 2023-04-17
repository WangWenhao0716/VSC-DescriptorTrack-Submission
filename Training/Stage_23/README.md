# Stage 2 and 3

## Prepare:
1. Build the support dataset:
```
mkdir /raid/VSC/data/training_images_all
cp -r /raid/VSC/data/training_images /raid/VSC/data/training_images_all/
```

2. Move the suppprt model ```train_v1_50C_s27_512_twolosses_m0.6_all_bw.pth.tar``` to here.

3. Move the support file ```pull_gt_all_shuffle_q.pickle``` and ```pull_gt_all_shuffle_r.pickle``` to here.


## Train:

```
bash train_swin.sh
bash train_vit.sh
bash train_t2t.sh
```
Plea
```
bash train_50.sh
bash train_50X.sh
bash train_50SK.sh
```

## Clean:

```
bash clean_swin.sh
bash clean_vit.sh
bash clean_t2t.sh
bash clean_50.sh
bash clean_50X.sh
bash clean_50SK.sh
```

## Conclusion:
Finally, we will get $6$ models for test:
1. ```train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

2. ```train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

3. ```train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

4. ```train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```

5. ```train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```

6. ```train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```
