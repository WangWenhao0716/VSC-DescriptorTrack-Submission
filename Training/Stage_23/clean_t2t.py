import torch
mod = torch.load('logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize/checkpoint_9_ema.pth.tar',map_location='cpu')
torch.save(mod['state_dict'], \
'logs/train_v1_s3_all_bw/t2t_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_resize_FIN.pth.tar')
