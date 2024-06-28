#coding:utf-8
import os
import threading

def process_video(video_path, frame_path, subset, vid, semaphore):
    with semaphore:
        vid_name = vid.split('.')[0]
        dst_frame_path = os.path.join(frame_path, subset, vid_name)
        os.makedirs(dst_frame_path, exist_ok=True)
        single_video_path = os.path.join(video_path, subset, vid)
        ffmpeg_cmd = r"ffmpeg -i '%s' -f image2 -vf 'fps=25,scale=-1:256:flags=bilinear' -loglevel quiet '%s'" % (single_video_path, dst_frame_path + '/img_%05d.jpg')
        os.system(ffmpeg_cmd)
        print(f"Processed {vid}")

def process_subset(video_path, frame_path, subset, num_threads):
    num = 1
    video_list = os.listdir(os.path.join(video_path, subset))
    print(video_list)
    os.makedirs(os.path.join(frame_path, subset), exist_ok=True)
    total_num = len(video_list)

    semaphore = threading.Semaphore(num_threads)
    threads = []
    for vid in video_list:
        print(f"[PROCESS] ({num}/{total_num}) {subset}: {vid}")
        num += 1
        thread = threading.Thread(target=process_video, args=(video_path, frame_path, subset, vid, semaphore))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def process_all_subsets(video_path, frame_path, subsets, num_threads=8):
    for subset in subsets:
        process_subset(video_path, frame_path, subset, num_threads)

video_path = './data/mmad_video'
frame_path = './data/mmad_video/frames'

subsets = ["train", "val", "test"]  
process_all_subsets(video_path, frame_path, subsets)