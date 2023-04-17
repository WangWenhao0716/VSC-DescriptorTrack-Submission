mkdir -p ./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN_test
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /raid/VSC/data/test/reference_one_second_imageio_v3_padcrop \
      --o ./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN_test/reference_v1_padcrop.hdf5 \
      --model swin_base  --GeM_p 3 --bw \
      --checkpoint train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar --imsize 224 
