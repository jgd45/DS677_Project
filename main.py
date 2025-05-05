import pandas as pd
import os
import cv2
import numpy as np
import sys
import random
import shutil
import yaml
sys.path.append('/Users/jordandavis/Documents/GitHub/DS677_Project/ultralytics/')
from ultralytics import YOLO

# ————— CONFIG —————
base = '/Users/jordandavis/Documents/DS677/Dataset'
train_imgs = os.path.join(base, 'train', 'images')
train_lbls = os.path.join(base, 'train', 'labels')
subset_dir = os.path.join(base, 'subset')       # ← where subset lives
num_samples = 200                                # ← how many images you want

# ————— MAKE SUBSET DIRS —————
for split in ['train']:
    for kind in ['images','labels']:
        d = os.path.join(subset_dir, split, kind)
        os.makedirs(d, exist_ok=True)

# ————— SAMPLE & COPY —————
all_imgs = [f for f in os.listdir(train_imgs) 
            if f.lower().endswith(('.jpg','.png'))]
chosen = random.sample(all_imgs, num_samples)

for fn in chosen:
    # copy image
    shutil.copy(
        os.path.join(train_imgs, fn),
        os.path.join(subset_dir, 'train', 'images', fn)
    )
    # copy label (same name but .txt)
    lbl = fn.rsplit('.',1)[0] + '.txt'
    src_lbl = os.path.join(train_lbls, lbl)
    dst_lbl = os.path.join(subset_dir, 'train', 'labels', lbl)
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, dst_lbl)

# ————— WRITE data.yaml —————
cfg = {
    'path': subset_dir,
    'train': 'train/images',
    'val':   'train/images',   # or point to your real val set
    'test':  'train/images',   # or point to your real test set
    'nc':    2,
    'names': ['bird','drone']
}

with open(os.path.join(subset_dir, 'data.yaml'), 'w') as f:
    yaml.safe_dump(cfg, f)

# ————— TRAIN —————
model = YOLO('yolov10n.yaml')   # or whatever variant you like
model.train(
    data=os.path.join(subset_dir,'data.yaml'),
    epochs=30,
    imgsz=640,
    batch=16,
    project='bird-drone-subset',
    name=f'subset{num_samples}',
    pretrained=True
)
results = model.val(data=os.path.join(subset_dir,'data.yaml'), plots=True)

# this returns [precision, recall, mAP50, mAP50-95]
p, r, mAP50, mAP5095 = results.box.mean_results()

print(f"Precision:      {p:.4f}")
print(f"Recall:         {r:.4f}")
print(f"mAP @ 0.50:     {mAP50:.4f}")
print(f"mAP @ 0.50–0.95: {mAP5095:.4f}")