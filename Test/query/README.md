## Prepare

Please download the required trained models and store them in ```./```:

1. [`best_20230101.pt`](https://drive.google.com/file/d/1N5B0nek4wYFeLj-KZJz0SUfAkpKoDNUV/view?usp=share_link)

2. [`rotate_detect.pth`](https://drive.google.com/file/d/1lQRvr8t_y3Pexb9PDzH6RzQDwXzOqVcc/view?usp=share_link)


## Run
We assume the queries are stored as follows:
```
ROOT_DIRECTORY = "/code_execution/"
DATA_DIRECTORY = "/data/"
QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY + "query/"
OUTPUT_FILE = ROOT_DIRECTORY + "subset_query_descriptors.npz"
QUERY_SUBSET_FILE = DATA_DIRECTORY + "query_subset.csv"
```

By running:
```
conda run --no-capture-output -n condaenv python main.py
```
You can get the subset query features.

## The way to get the `submission.zip`

First get the ```timm-0.4.12-py3-none-any.whl``` by: 
```
wget https://files.pythonhosted.org/packages/90/fc/606bc5cf46acac3aa9bd179b3954433c026aaf88ea98d6b19f5d14c336da/timm-0.4.12-py3-none-any.whl
```

Then change the file path in `Run` Section, you can get the query features. An example:

Change from 

```
OUTPUT_FILE = ROOT_DIRECTORY + "subset_query_descriptors.npz"
QUERY_SUBSET_FILE = DATA_DIRECTORY + "query_subset.csv"
```
to 
```
OUTPUT_FILE = ROOT_DIRECTORY + "query_descriptors.npz"
QUERY_SUBSET_FILE = DATA_DIRECTORY + "query.csv"
```

Given the copied, downloaded, and generated files, together with the code in this folder, we can get the `submission.zip` by:

```
zip -r submission.zip ./*
```

