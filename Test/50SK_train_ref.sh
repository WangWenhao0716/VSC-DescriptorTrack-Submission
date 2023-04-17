mkdir -p ./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /raid/VSC/data/train/reference_one_second_imageio_v3 \
      --o ./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/reference_v1.hdf5 \
      --model 50SK  --GeM_p 3 --bw \
      --checkpoint train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar --imsize 256 
