#!/usr/bin/python
# coding:utf-8

import csv
import json
import os
import cv2
from tqdm import tqdm

annotation_path = './data/mmad_video/'
all_sets = ['train', 'val', 'test']

failed_videos = {}
dataset = {}
for sub_set in all_sets:
    print(sub_set)
    with open(os.path.join(annotation_path, 'anns', sub_set + '.csv'), 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in tqdm(csv_reader):
            video_id = row[1]
            start_frame = int(row[2])
            end_frame = int(row[3])
            label = int(row[4])
            assert 0 <= label < 52, f"Label {video_id} at index is out of range"
            total_frames = int(row[5])

            video_path = os.path.join(annotation_path, os.path.join(sub_set, video_id + '.mp4'))
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Open video failed: {video_path}")
                failed_videos.append(video_id)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if video_id in dataset:
                actions = dataset[video_id]["actions"]
                actions.append((label, round(start_frame / fps, 4), round(end_frame / fps, 4)))
                dataset[video_id] = {
                    'subset': sub_set,
                    'duration': round(total_frames / fps, 4),
                    'actions': actions
                }
            else:
                actions = [(label, round(start_frame / fps, 4), round(end_frame / fps, 4))]
                dataset[video_id] = {
                    'subset': sub_set,
                    'duration': round(total_frames / fps, 4),
                    'actions': actions
                }
print(len(dataset))
json_path = os.path.join('./datasets/', 'mmad.json')

with open(json_path, "w") as file:
    json.dump(dataset, file, indent=4, separators=(',', ':'))
print(failed_videos)
