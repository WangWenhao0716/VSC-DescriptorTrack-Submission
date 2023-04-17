# Generate

## Download the datasets

1. The training set of [DISC21](https://ai.facebook.com/datasets/disc21-downloads/).

2. The [fonts files](https://drive.google.com/file/d/17NabnySxASqAvYEscW-b-YTA0SU4J-ys/view?usp=share_link) from [Augly](https://github.com/facebookresearch/AugLy).

3. The [emoji files](https://drive.google.com/file/d/1--Sx8uHthQAVLrDdR_C4ibveT_nGJunC/view?usp=share_link) from [Augly](https://github.com/facebookresearch/AugLy).

4. The training set for [VSC](https://www.drivendata.org/competitions/group/meta-video-similarity/).

We assume that, after downloading, we have:

1. ```/raid/VSC/data/training_images/``` contains all the training images of DISC21.

2. ```/raid/VSC/data/training_images_9/``` contains the training images of DISC21 which end with $9$, e.g. ```T000009.jpg```.

3. ```/raid/VSC/data/fonts/``` contains the fonts files from Augly.

4. ```/raid/VSC/data/emoji/``` contains the emoji files from Augly.

5. ```/raid/VSC/data/train/reference``` contains the reference images for training.

6. ```/raid/VSC/data/train/query``` contains the query images for training.

7. ```/raid/VSC/data/train/train_matching_ground_truth.csv``` is the ground truth file for the training set.


## Generate the training data from DISC21

We use a server which can run 250 processes at once. Hope your server is also powerful!

```
bash train_v1_s27_all.sh
bash train_v1_s3_all.sh
bash train_v1_s27_all_bw.sh
bash train_v1_s3_all_bw.sh
```

A known issue brought by Augly package is:
```
Traceback (most recent call last):
  File "train_v1_s27_all.py", line 462, in <module>
    image_q = transform_q(image)
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/torchvision/transforms/transforms.py", line 60, in __call__
    img = t(img)
  File "train_v1_s27_all.py", line 190, in __call__
    y_pos = y_pos)(x)
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/augly/image/transforms.py", line 64, in __call__
    return self.apply_transform(image, metadata, bboxes, bbox_format)
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/augly/image/transforms.py", line 1490, in apply_transform
    bbox_format=bbox_format,
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/augly/image/functional.py", line 1606, in overlay_text
    font=font,
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/PIL/ImageDraw.py", line 463, in text
    draw_text(ink)
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/PIL/ImageDraw.py", line 418, in draw_text
    **kwargs,
  File "/home/wangwenhao/anaconda3/envs/ISC/lib/python3.7/site-packages/PIL/ImageFont.py", line 677, in getmask2
    text, im.id, mode, direction, features, language, stroke_width, ink
OSError: raster overflow
```
You should re-run the ```bash xxx.sh``` until the number of images in ```/raid/VSC/images/train_v1_s27_all/train_v1_s27_all/``` or ```/raid/VSC/images/train_v1_s3_all/train_v1_s3_all/``` equals to $2,000,000$. If you know how to fix this random bug, please let me know (wangwenhao0716@gmail.com).

## Generate the training data from VSC

1. Transfer the query videos into images using ```imageio```:
```
bash video2images_query.sh
```

2. Transfer the reference videos into images using ```imageio```:
```
bash video2images_ref.sh
```

3. Select one image from each video to train:
```
bash select_training_reference.sh
```

4. Generate the training set (from reference images) with $40,109 \times 20 = 802,180$ images:
```
bash train_v1_s3_r_all.sh
```

5. Generate the training pairs (from reference images and query images) with $31,425$ positive pairs:
```
python generate_pair.py
python shuffle.py
```
In this step, we also generate ```pull_gt_all_shuffle_q.pickle``` and ```pull_gt_all_shuffle_r.pickle```.

