CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_com_bw.py \
-ds train_v1_s3_all_bw -a T2T_vit_14 --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /raid/VSC/images/ \
--logs-dir logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_same_resize \
--height 224 --width 224

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_com_tune_bw.py \
-ds train_v1_s3_all_bw -a T2T_vit_14 --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 10 \
--lr 0.00035 --iters 8000 --epochs 10 \
--data-dir /raid/VSC/images/ \
--logs-dir logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_same_tune_resize \
--height 224 --width 224 \
--resume logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_same_resize/checkpoint_24_ema.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_com_tune_bw_gt_ng_cls.py \
-ds train_v1_s3_all -ds-small train_v1_s3_r1_all -a T2T_vit_14_double --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 10 \
--lr 0.00035 --iters 8000 --epochs 10 \
--data-dir /raid/VSC/images/ \
--logs-dir logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize \
--height 224 --width 224 \
--resume logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_same_tune_resize/checkpoint_9_ema.pth.tar
