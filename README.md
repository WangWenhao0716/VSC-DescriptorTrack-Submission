# VSC-DescriptorTrack-Submission ($2^{nd}$ Solution)
The codes and related files to reproduce the results for [Video Similarity Challenge Descriptor Track](https://www.drivendata.org/competitions/101/meta-video-similarity-descriptor/) (CVPR2023).

## Required dependencies
To begin with, you should install the packages according to the `environment.yaml` file in this directory. Then install the `GNU parallel` by ```sudo apt-get install parallel``` (For Ubuntu). The minimum requirment for training is 8 Nvidia A100 40G GPUs, and for reference, you should have 1 Nvidia V100 16G GPU at least. 

## Pre-trained models

We use $7$ ImageNet-pre-trained models. Please download them from the provided links as below:

0. CotNet50: [download_from_original_repo](https://drive.google.com/file/d/1SR5ezIu7LN943zHaUh4mC0ehxBVMqtfv/view); The original project is [CoTNet](https://github.com/JDAI-CV/CoTNet).

1. ResNet50: No need to download manually;

2. SKNet50: [Google Drive](https://drive.google.com/file/d/1h6V3zhWGB_kCAIuXimPWKZnLlu21T7T8/view?usp=share_link); The original project is [SKNet-PyTorch](https://github.com/developer0hye/SKNet-PyTorch/tree/master). Because it does not include the pre-trained models, please download the ```sknet.py``` file in this repository and follow the instruction in the ```Pretrain``` folder.

3. Resnext50_32x4d: No need to download manually;

4. ViT: No need to download manually;

5. Swin: No need to download manually;

6. T2T: [download_from_original_repo](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar); The original project is [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT).

## Training

For training, we propose Feature-Compatible Progressive Learning (FCPL). Please refer to the ```Training``` folder for more details.

## Test

The ```Test``` folder shows the submitted code to extract query features plus the code to extract reference features and normalization features.

## Citation

```
@article{wang2023feature,
  title={Feature-compatible Progressive Learning for Video Copy Detection},
  author={Wang, Wenhao and Sun, Yifan and Yang, Yi},
  journal={arXiv preprint arXiv:2304.10305},
  year={2023}
}
```

## Bug finding
Please raise an issue or send an email to wangwenhao0716@gmail.com if a bug exists. Thanks!

