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


## Generate the training data

We use a server which can run 250 processes at once. Hope your server is also powerful!

```
bash train_v1_s27_all.sh
bash train_v1_s3_all.sh
bash train_v1_s27_all_bw.sh
bash train_v1_s3_all_bw.sh
```


