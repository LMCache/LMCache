import os
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
def select_videos(input_directory, output_json):
    selected_videos = []
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                try:
                    clip = VideoFileClip(video_path)
                    duration = clip.duration
                    clip.close()
                    
                    if 60 <= duration <= 120:
                        category = os.path.basename(root)  # Assuming category is the folder name
                        selected_videos.append({
                            "video_path": video_path,
                            "category": category,
                            "duration": duration
                        })
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    
    with open(output_json, 'w') as json_file:
        json.dump(selected_videos, json_file, indent=4)

def save_selected_videos_from_txt(txt_file, output_json):
    selected_videos = []
    
    with open(txt_file, 'r') as file:
        video_paths = file.readlines()
    
    for video_path in video_paths:
        video_path = video_path.strip()
        video_path = video_path.replace('/home/users/ntu/yulin001/scratch/', '/root/workspace/dataset/Anomaly-Detection-Dataset/')
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            video_path = video_path.replace('/root/workspace/dataset/Anomaly-Detection-Dataset/dataset/', 'dataset/')
            # dataset/Fighting028_x264.mp4, 抽取 category 为 Fighting，后续的数字和_x264.mp4 去掉，第一个不一定是 0
            category = ''.join([c for c in video_path.split('/')[1] if not c.isdigit()]).replace('_x264.mp4', '')
            category = category.replace('_x.mp','').lower()
            if 'normal' in category:
                category = 'normal'
            selected_videos.append({
                "video_path": video_path,
                "category": category,
                "duration": duration
            })
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    # with open(output_json, 'w') as json_file:
    #     json.dump(selected_videos, json_file, indent=4)
    # print all categories
    categories = set([video['category'] for video in selected_videos])
    print("Categories:", categories)    

# Example usage
# input_directory = 'path/to/your/video/dataset'
# output_json = 'selected_videos.json'
# select_videos(input_directory, output_json)

save_selected_videos_from_txt('/root/workspace/lmcache_multimodal/scripts_test_video/datasets/small_dataset.txt', '/root/workspace/lmcache_multimodal/scripts_test_video/datasets/small_dataset.json')