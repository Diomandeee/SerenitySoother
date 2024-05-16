import os
import glob
import re

def natural_sort_key(s: str, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def parse_directory(base_path: str) -> dict:
    data = {}
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            data[dir_name] = {
                'scene_descriptions': [],
                'sections': [],
                'images': [],
                'audios': []
            }
            scene_description_files = sorted(glob.glob(os.path.join(dir_path, 'scene_descriptions', '*.txt')), key=natural_sort_key)
            for i, file in enumerate(scene_description_files):
                data[dir_name]['scene_descriptions'].append((i, read_file(file)))
            section_files = sorted(glob.glob(os.path.join(dir_path, 'sections', '*.txt')), key=natural_sort_key)
            for i, file in enumerate(section_files):
                data[dir_name]['sections'].append((i, read_file(file)))
            image_files = sorted(glob.glob(os.path.join(dir_path, 'images', '*.png')), key=natural_sort_key)
            for i, file in enumerate(image_files):
                data[dir_name]['images'].append((i, file))
            audio_files = sorted(glob.glob(os.path.join(dir_path, 'audios', '*.mp3')), key=natural_sort_key)
            for i, file in enumerate(audio_files):
                data[dir_name]['audios'].append((i, file))
    return data

