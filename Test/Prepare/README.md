# The training process for the auxiliary models.

## Get ```deblack_20230107.pt```


## Get ```best_20230101.pt```

## Get ```rotate_detect.pth```

Please first enter the ```rotate_detect``` folder by ```cd rotate_detect```, and then follow the below instructions.

1. Download the pre-trained models:
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth
```

2. The training dataset is auto-generated (by rotating images from  DISC21 90, 180, 270 degrees), you can directly download from [here](), and unzip by:
```
tar -xvf rotate_images.tar
```

3. Assume we have $8$ A100 GPUs, and the images are stored in ```/raid/VSC/images/rotate_images```; we can train the model by

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_small_patch4_window7_224.yaml --data-path /raid/VSC/images/rotate_images --batch-size 128 \
--pretrained swin_small_patch4_window7_224_22k.pth
```

4. After training, we use the final checkpoint as the ```rotate_detect.pth``` by: 

```
cp output/swin_small_patch4_window7_224/default/ckpt_epoch_299.pth rotate_detect.pth
```
