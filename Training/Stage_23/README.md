# Stage 2 and 3

## Prepare:
1. Build the support dataset:
```
mkdir /raid/VSC/data/training_images_all
cp -r /raid/VSC/data/training_images /raid/VSC/data/training_images_all/
```

2. Move the suppprt model ```train_v1_50C_s27_512_twolosses_m0.6_all_bw.pth.tar``` to ```./```.

3. Move the support files ```pull_gt_all_shuffle_q.pickle``` and ```pull_gt_all_shuffle_r.pickle``` to ```./```.

4. Move the pre-trained models ```81.5_T2T_ViT_14.pth.tar``` and ```sknet_imagenet_pretrained.pth.tar``` to ```./logs/pretrained/```.


## Train:

```
bash train_swin.sh
bash train_vit.sh
bash train_t2t.sh
```
Please change ```dg/trainers_cos_ema_tune_gt_ng_cls.py```:

Line 127 from 
```
loss = loss_ce + loss_ce_1 + loss_ce_small + loss_ce_small_1 + 100 * loss_sp + 1 * (loss_gt - loss_ng)
```
to 
```
loss = loss_ce + loss_ce_1 + loss_ce_small + loss_ce_small_1 + 100 * loss_sp + 2 * (loss_gt - loss_ng)
```
before the training of CNN-based methods.

```
bash train_50.sh
bash train_50X.sh
bash train_50SK.sh
```

## Clean:

```
python clean_swin.py
python clean_vit.py
python clean_t2t.py
python clean_50.py
python clean_50X.py
python clean_50SK.py
```

## Conclusion:
Finally, we will get $6$ models for test:
1. ```train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

2. ```train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

3. ```train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar```

4. ```train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```

5. ```train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```

6. ```train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar```
