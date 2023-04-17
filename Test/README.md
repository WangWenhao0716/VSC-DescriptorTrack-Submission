# Test
You can directly download and unzip our [submitted file](https://drive.google.com/file/d/1--OST0kIrNsLBa7MCArztV9u46pKjcip/view?usp=share_link) (including all the trained models, and all other necessary files), then by running
```
conda run --no-capture-output -n condaenv python main.py
```
you will get the query features. 

However, because the file only includes extracted reference features and normalization features, we show how to get these two kinds of features here.

Assuming we have downloaded the training and test reference datasets, and stored as follows:

```
/raid/VSC/data/train/reference/
/raid/VSC/data/test/reference/
```

1. We first transform ```/raid/VSC/data/test/reference/``` into images using ```imageio``` by:

```
```
Note we have transformed the ```/raid/VSC/data/train/reference/``` into images in the training section.

2. 
