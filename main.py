import pandas as pd
import os
import cv2
import numpy as np
import sys
import random
import shutil
import yaml
sys.path.append('/home/ordan_avis/DS677_Project/ultralytics/')
from ultralytics import YOLO

# ————— CONFIG —————
base       = '/home/ordan_avis/DS677/Dataset'        # root of your full dataset
subset_dir = os.path.join(base, 'subset')            # where we’ll build the subset
splits     = ['train', 'valid', 'test']              # original split names
# number of images to sample per split
num_samples = {'train': 1000, 'valid': 500, 'test': 500}
classes    = ['bird', 'drone']

# file‐prefix → class mapping, per split
prefix_map = {
    'train': ('BT', 'DT'),   # train images start with BT (bird) or DT (drone)
    'valid': ('BV', 'DV'),   # valid images start with BV (bird) or DV (drone)
    'test':  ('BT', 'DT'),   # test uses the same prefixes as train
}

# ————— CLEAN & CREATE SUBSET ROOT —————
if os.path.exists(subset_dir):
    shutil.rmtree(subset_dir)
os.makedirs(subset_dir, exist_ok=True)

# ————— STRATIFIED SAMPLE & COPY —————
for split in splits:
    bird_pref, drone_pref = prefix_map[split]
    src_dir = os.path.join(base, split, 'images')
    dst_dir = os.path.join(subset_dir, split, 'images')
    os.makedirs(dst_dir, exist_ok=True)

    # list all image files
    all_imgs = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]

    # split into bird vs drone pools
    bird_pool  = [f for f in all_imgs if f.startswith(bird_pref)]
    drone_pool = [f for f in all_imgs if f.startswith(drone_pref)]
    per_class  = num_samples[split] // 2

    # sample up to per_class from each pool
    chosen = []
    chosen += random.sample(bird_pool,  min(per_class, len(bird_pool)))
    chosen += random.sample(drone_pool, min(per_class, len(drone_pool)))

    # if there’s any remainder, fill from the combined pool
    full_pool = bird_pool + drone_pool
    while len(chosen) < num_samples[split]:
        chosen.append(random.choice(full_pool))

    # copy the chosen images
    for fn in chosen:
        shutil.copy(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))

# ————— RESTRUCTURE FOR CLASSIFICATION —————
# move images into split/{bird,drone} based on prefixes
for split in splits:
    bird_pref, drone_pref = prefix_map[split]
    img_dir = os.path.join(subset_dir, split, 'images')

    # make class subfolders
    for cls in classes:
        os.makedirs(os.path.join(subset_dir, split, cls), exist_ok=True)

    # move each image into the appropriate class folder
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.jpg', '.png')):
            continue
        if fn.startswith(bird_pref):
            cls = 'bird'
        elif fn.startswith(drone_pref):
            cls = 'drone'
        else:
            # fallback: treat any other prefix as drone
            cls = 'drone'
        shutil.move(
            os.path.join(img_dir, fn),
            os.path.join(subset_dir, split, cls, fn)
        )

    # remove the now‐empty images folder
    shutil.rmtree(img_dir)

# ————— RENAME valid → val —————
# classification mode expects 'train', 'val', 'test'
os.rename(
    os.path.join(subset_dir, 'valid'),
    os.path.join(subset_dir, 'val')
)

print("Subset ready under:", subset_dir)
print("Splits:", os.listdir(subset_dir))  # should list ['train', 'val', 'test']

# ————— TRAIN AS CLASSIFIER —————
# Ensure your yolov10n.yaml has the built-in Classify head:
# head:
#   - [-1, 1, Classify, [nc]]
# and at top: nc: 2

model = YOLO('yolov10n.yaml', task='classify')
run_name = f"subset_{num_samples['train']}_{num_samples['valid']}_{num_samples['test']}"

model.train(
    data=subset_dir,       # point to the root folder
    epochs=100,
    imgsz=224,             # smaller for classification
    batch=16,              # reduce if you hit OOM
    project='bird-drone-subset',
    name=run_name,
    pretrained=True
)


# ————— VALIDATE & REPORT —————
metrics = model.val(data=subset_dir, plots=True)
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")