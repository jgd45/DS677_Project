import pandas as pd
import os
import cv2
import numpy as np
import sys
import random
import shutil

sys.path.append('/Users/jordandavis/Documents/GitHub/DS677_Project/ultralytics/')
from ultralytics import YOLO

#https://www.kaggle.com/datasets/stealthknight/bird-vs-drone
base_dir = '/Users/jordandavis/Documents/DS677/Dataset/'
test_dir = base_dir + 'test/'
test_images_dir = test_dir + 'images/'
image_paths = []
image_labels = []
destination_dir = '/Users/jordandavis/Documents/DS677/Dataset/random'


#function to choose random files from folders and place them in a new folder
def random_file_choose(source_dir, destination_dir, num_files):
    files = [f for f in os.listdir(source_dir) 
             if os.path.isfile(os.path.join(source_dir, f))]
    
    selected_files = random.sample(files,num_files)

    os.makedirs(destination_dir, exist_ok=True)

    for file in selected_files:
        shutil.copy(os.path.join(source_dir,file), os.join(destination_dir,file))
    return files


files = random_file_choose(test_images_dir, base_dir,10)


model = YOLO('yolov8n.pt')  # Or yolov8s.pt or yolov8m.pt depending on your compute

model.train(data='/Users/jordandavis/Documents/DS677/Dataset/data.yaml', epochs=50, imgsz=640)
