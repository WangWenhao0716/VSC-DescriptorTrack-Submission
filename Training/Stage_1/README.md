# Training Stage 1

1. You should move the pre-trained model ```cotnet50.pth.tar``` to ```./logs/pretrained/```.

2. In this stage, we train a base model (cotnet-50), by:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema.py \
-ds train_v1_s27_all_bw -a cotnet50 --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /raid/VSC/images/ \
--logs-dir logs/train_v1_s27_all_bw/50C_two_losses_m0.6 \
--height 256 --width 256
```

3. Clean the model:

```
import torch
mod = torch.load('logs/train_v1_s27_all_bw/50C_two_losses_m0.6/checkpoint_24_ema.pth.tar',map_location='cpu')
torch.save(mod['state_dict'], 'logs/train_v1_s27_all_bw/50C_two_losses_m0.6/train_v1_50C_s27_512_twolosses_m0.6_all_bw.pth.tar')
```

The model ```train_v1_50C_s27_512_twolosses_m0.6_all_bw.pth.tar``` is regarded as the base model for the next stages.
