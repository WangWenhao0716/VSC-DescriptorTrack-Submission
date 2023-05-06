import ffmpeg
import os
import pickle
import pandas as pd

DATA_DIRECTORY = "/data/"
QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY + "query/"
QUERY_SUBSET_FILE = DATA_DIRECTORY + "query_subset.csv"
query_subset = pd.read_csv(QUERY_SUBSET_FILE)
query_subset_video_ids = list(query_subset.video_id.values.astype("U"))

for i in range(len(query_subset_video_ids)): # add .mp4
    if '.mp4' not in query_subset_video_ids[i]:
        query_subset_video_ids[i] = query_subset_video_ids[i] + '.mp4'

path = QRY_VIDEOS_DIRECTORY
ls = query_subset_video_ids

save_final = []
for i in range(len(ls)):
    if i%100==0:
        print(i)
    pr = ffmpeg.probe(path + ls[i])["streams"]
    if len(pr) > 1:
        print("Wow!!!")
        save_final.append(ls[i])
    else:
        if 'sample_aspect_ratio' in pr[0].keys():
            print("Haha!!!")
            save_final.append(ls[i])

save_final = [s.split('.')[0] for s in save_final] # no .mp4

tranditional_filter = save_final
for i in tranditional_filter:
    assert 'mp4' not in i
        
with open('tranditional_filter.pkl', 'wb') as f:
    pickle.dump(tranditional_filter, f)
    