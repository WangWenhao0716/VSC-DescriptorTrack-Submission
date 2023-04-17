from isc.io import read_descriptors, write_hdf5_descriptors
import numpy as np
import faiss

name_r, resnet_r = \
read_descriptors(['./feature/train_v1_50_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN_test/reference_%d_v1_padcrop.hdf5'%i for i in range(0,25)])

name_r, resneXt_r = \
read_descriptors(['./feature/train_v1_50X_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN_test/reference_%d_v1_padcrop.hdf5'%i for i in range(0,25)])

name_r, sknet_r = \
read_descriptors(['./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN_test/reference_%d_v1_padcrop.hdf5'%i for i in range(0,25)])

name_r, vit_r = \
read_descriptors(['./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN_test/reference_%d_v1_padcrop.hdf5'%i for i in range(0,25)])

name_r, t2t_r = \
read_descriptors(['./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN_test/reference_%d_v1_padcrop.hdf5'%i for i in range(0,25)])

name_r, swin_r = \
read_descriptors(['./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN_test/reference_%d_v1_padcrop.hdf5'%i for i in range(0,25)])

select_features_r = []
for i in range(len(name_r)):
    if i%1000==0:
        print(i)
    f_2 = resnet_r[i].reshape((512,1))
    f_3 = resneXt_r[i].reshape((512,1))
    f_7 = sknet_r[i].reshape((512,1))
    f_11 = vit_r[i].reshape((512,1))
    f_12 = t2t_r[i].reshape((512,1))
    f_13 = swin_r[i].reshape((512,1))
    con = np.concatenate((f_2, f_3, f_7, f_11, f_12, f_13),axis=1)
    new = np.mean(con, axis=1).reshape(1,512)
    select_features_r.append(new)
    
feature_r = np.concatenate(select_features_r,axis=0)
faiss.normalize_L2(feature_r)
write_hdf5_descriptors(feature_r, name_r, 'ensemble_6_r_512_crop_rotate_bw_gt_ng_1_cls_FIN.hdf5')

db_image_ids, xb = read_descriptors('ensemble_6_r_512_crop_rotate_bw_gt_ng_1_cls_FIN.hdf5')
xb = xb[:,:-1]
xb_1 = np.hstack((xb, np.ones((len(xb), 1), dtype='float32')))
assert xb_1.shape[1] == 512
db_image_ids_clean = [r.split('_')[0] for r in db_image_ids]
db_timestamps = np.array([[0,0]]*len(db_image_ids_clean), dtype=np.float32)
np.savez(
    "reference_descriptors.npz",
    video_ids = db_image_ids_clean,
    timestamps = db_timestamps,
    features = xb_1,
)



