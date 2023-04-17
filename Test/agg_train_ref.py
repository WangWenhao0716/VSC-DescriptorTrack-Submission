from isc.io import read_descriptors, write_hdf5_descriptors
import numpy as np
import faiss

name_t, resnet_t = \
read_descriptors(['./feature/train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/reference_%d_v1.hdf5'%i for i in range(0,25)])

name_t, resneXt_t = \
read_descriptors(['./feature/train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/reference_%d_v1.hdf5'%i for i in range(0,25)])

name_t, sknet_t = \
read_descriptors(['./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/reference_%d_v1.hdf5'%i for i in range(0,25)])

name_t, vit_t = \
read_descriptors(['./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/reference_%d_v1.hdf5'%i for i in range(0,25)])

name_t, t2t_t = \
read_descriptors(['./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/reference_%d_v1.hdf5'%i for i in range(0,25)])

name_t, swin_t = \
read_descriptors(['./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN/reference_%d_v1.hdf5'%i for i in range(0,25)])

select_features_t = []
for i in range(len(name_t)):
    if i%1000==0:
        print(i)
    f_2 = resnet_t[i].reshape((512,1))
    f_3 = resneXt_t[i].reshape((512,1))
    f_7 = sknet_t[i].reshape((512,1))
    f_11 = vit_t[i].reshape((512,1))
    f_12 = t2t_t[i].reshape((512,1))
    f_13 = swin_t[i].reshape((512,1))
    con = np.concatenate((f_2, f_3, f_7, f_11, f_12, f_13),axis=1)
    new = np.mean(con, axis=1).reshape(1,512)
    select_features_t.append(new)
    
feature_t = np.concatenate(select_features_t,axis=0)
faiss.normalize_L2(feature_t)
write_hdf5_descriptors(feature_t, name_t, 'ensemble_6_t_512_crop_rotate_bw_gt_ng_1_cls_FIN.hdf5')
    
