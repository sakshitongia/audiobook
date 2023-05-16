import cv2
import moviepy.editor as mymovie
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips
import shutil
import os
def result():
    clips = []
    folder_path = 'static/final/'
    files = os.listdir(folder_path) # here I need all the paths in static/final output
    print('files',files)
    for i in range(len(files)):
        print(i)
        if files[i][-1] == '4':
            clip = VideoFileClip(folder_path+files[i])
            clips.append(clip)
    print(clips)
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("static/Final1/output_comic.mp4", fps=24, codec='libx264', audio_codec='aac',
                               remove_temp=True)

    def deleting_files(temp_file_paths):
        for filename in os.listdir(temp_file_paths):
            file_path = os.path.join(temp_file_paths, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    temp_file_paths = "static/final"
    deleting_files(temp_file_paths)
