import os
import glob
from typing import Dict, List
import re

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def natural_sort_key(s: str, _nsre=re.compile('([0-9]+)')):
    """
    Natural sort key function to sort strings containing numbers in human order.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def parse_directory(base_path: str) -> Dict[str, Dict[str, List[str]]]:
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
            
            # Read scene descriptions
            scene_description_files = sorted(glob.glob(os.path.join(dir_path, 'scene_descriptions', '*.txt')), key=natural_sort_key)
            for file in scene_description_files:
                data[dir_name]['scene_descriptions'].append(read_file(file))
            
            # Read sections
            section_files = sorted(glob.glob(os.path.join(dir_path, 'sections', '*.txt')), key=natural_sort_key)
            for file in section_files:
                data[dir_name]['sections'].append(read_file(file))
            
            # Read images
            image_files = sorted(glob.glob(os.path.join(dir_path, 'images', '*.png')), key=natural_sort_key)
            for file in image_files:
                data[dir_name]['images'].append(file)
            
            # Read audios
            audio_files = sorted(glob.glob(os.path.join(dir_path, 'audios', '*.mp3')), key=natural_sort_key)
            for file in audio_files:
                data[dir_name]['audios'].append(file)
    
    return data

def main():
    base_path = 'data'
    parsed_data = parse_directory(base_path)
    
    for key, value in parsed_data.items():
        print(f"ID: {key}")
        print("Scene Descriptions:")
        for i, desc in enumerate(value['scene_descriptions']):
            print(f"  {i+1}. {desc}")
        print("Sections:")
        for i, section in enumerate(value['sections']):
            print(f"  {i+1}. {section}")
        print("Images:")
        for i, image in enumerate(value['images']):
            print(f"  {i+1}. {image}")
        print("Audios:")
        for i, audio in enumerate(value['audios']):
            print(f"  {i+1}. {audio}")

if __name__ == "__main__":
    main()
