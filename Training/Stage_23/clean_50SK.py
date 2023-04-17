import torch
mod = torch.load('logs/train_v1_s3_all_bw/50SK_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls/checkpoint_9_ema.pth.tar',map_location='cpu')
torch.save(mod['state_dict'], \
'logs/train_v1_s3_all_bw/50SK_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar')
