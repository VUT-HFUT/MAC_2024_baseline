#coding:utf-8
import os
import json
import numpy as np
import pandas as pd
import zipfile


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

eval_set = "test" # or val

eval_results = './outputs/results_{}.csv'.format(eval_set)
zip_file_path = './online_eval/prediction.csv'
submission_file_path = './online_eval/submission.zip'

df = pd.read_csv(eval_results)

new_columns = {
    't-start': 't_start',
    't-end': 't_end',
    'score': 'score',
    'label': 'class',
    'video-id': 'video_id'
}

# rename columns
df.rename(columns=new_columns, inplace=True)

new_order = ['video_id', 't_start', 't_end', 'class', 'score'] 
frame_dict = load_json('./datasets/mmad_frames.json')
annotations = load_json('./datasets/mmad.json')

update_df = df[new_order]
update_df.to_csv('./online_eval/prediction.csv')

# creat zip file for online submission
with zipfile.ZipFile(submission_file_path, 'w') as zipf:
    zipf.write(zip_file_path, os.path.basename(zip_file_path))
