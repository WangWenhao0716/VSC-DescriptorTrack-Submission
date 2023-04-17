# VSC-DescriptorTrack-Submission
The codes and related files to reproduce the results for Video Similarity Challenge Descriptor Track.

## Required dependencies
To begin with, you should follow the [official environment](https://github.com/drivendataorg/meta-vsc-descriptor-runtime/blob/main/runtime/environment-gpu.yml) to install the packages. Please install timm by ```pip install timm==0.4.12```. Note that some unimportant packages may be missing, please install them using pip directly when an error occurs. The minimum requirment for training is 4 Nvidia A100 40G GPUs, and for reference, you should have 1 Nvidia V100 16G GPU at least. 

## Pre-trained models

We use $7$ ImageNet-pre-trained models. Please download them from the provided links as below:

0. CotNet50: [Google Drive](https://drive.google.com/file/d/1-CVORVrELHFph45VNgAJmz_Fg8XHCxO1/view?usp=share_link);

1. ResNet50: No need to download manually;

2. SKNet50: [Google Drive](https://drive.google.com/file/d/1h6V3zhWGB_kCAIuXimPWKZnLlu21T7T8/view?usp=share_link);

3. Resnext50_32x4d: No need to download manually;

4. ViT: No need to download manually;

5. Swin: No need to download manually;

6. T2T: [Google Drive](https://drive.google.com/file/d/1-BdI3QKFAsYsv9Zd2GtaXefHqBFE2Dad/view?usp=share_link).

## Training

For training, we propose Feature-Compatible Progressive Learning (FCPL). Please refer to the ```Training``` folder for more details.

## Test

The ```Test``` folder shows the submitted code to extract query features plus the code to extract reference features and normalization features.

## Citation

```
@article{wang2023feature,
  title={Feature-compatible Progressive Learning for Video Copy Detection},
  author={Wang, Wenhao and Sun, Yifan and Yang, Yi},
  journal={arXiv preprint arXiv:2304.xxxxx},
  year={2023}
}
```


