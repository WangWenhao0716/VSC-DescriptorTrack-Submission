### Pretrain

To pre-train the SkNet-50, we assume you has $8$ A100 GPUs, and the ImageNet dataset is stored in `/raid/`:
```
python main.py -a sknet50 --dist-url 'tcp://127.0.0.1:8899' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 /raid/ILSVRC2012_RAW_PYTORCH
```

After pre-training, you should perform: 
```
cp model_best.pth.tar sknet_imagenet_pretrained.pth.tar
```
