import pandas as pd
from ultralytics import YOLO
import os
import cv2
import numpy as np
import sys
#data = pd.read
base_dir = '/Users/jordandavis/Documents/DS677/Dataset/'
test_dir = base_dir + 'test/'
image_paths = []
image_labels = []

model = YOLO('yolov8n.pt')  # Or yolov8s.pt or yolov8m.pt depending on your compute

model.train(data='/Users/jordandavis/Documents/DS677/Dataset/data.yaml', epochs=50, imgsz=640)
