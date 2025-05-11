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
num_samples = {'train': 1000, 'valid': 400, 'test': 400}
classes    = ['bird', 'drone']

# file‐prefix → class mapping, per split
prefix_map = {
    'train': ('BT', 'DT'),   # train images start with BT (bird) or DT (drone)
    'valid': ('BV', 'DV'),   # valid images start with BV (bird) or DV (drone)
    'test':  ('BT', 'DT'),   # test uses the same prefixes as train
}
task = 'detect'
if task == 'classify':
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


    model = YOLO('yolov10b.yaml', task='classify')
    run_name = f"subset_{num_samples['train']}_{num_samples['valid']}_{num_samples['test']}"

    model.train(
        data=subset_dir,       # point to the root folder
        epochs=100,
        imgsz=224,             # smaller for classification
        batch=8,              # reduce if you hit OOM
        project='bird-drone-subset',
        name=run_name,
        pretrained=True
    )


        # ————— VALIDATE & REPORT —————
    metrics = model.val(data=subset_dir, plots=True)
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")

else:
    # 1) Build a DETECTION subset in subset_detect/
    subset_detect_dir = os.path.join(base, 'subset_detect')
    if os.path.exists(subset_detect_dir):
        shutil.rmtree(subset_detect_dir)
    os.makedirs(subset_detect_dir, exist_ok=True)

    for split in splits:
        src_imgs = os.path.join(base, split, 'images')
        src_lbls = os.path.join(base, split, 'labels')
        dst_imgs = os.path.join(subset_detect_dir, split, 'images')
        dst_lbls = os.path.join(subset_detect_dir, split, 'labels')
        os.makedirs(dst_imgs, exist_ok=True)
        os.makedirs(dst_lbls, exist_ok=True)

        # sample images
        all_imgs = [f for f in os.listdir(src_imgs) if f.lower().endswith(('.jpg','.png'))]
        chosen   = random.sample(all_imgs, min(num_samples[split], len(all_imgs)))

        # copy both image and its label
        for fn in chosen:
            shutil.copy(os.path.join(src_imgs, fn), os.path.join(dst_imgs, fn))
            lbl = fn.rsplit('.',1)[0] + '.txt'
            src_lbl = os.path.join(src_lbls, lbl)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(dst_lbls, lbl))

    # rename 'valid' → 'val' so YOLO finds it
    os.rename(
        os.path.join(subset_detect_dir, 'valid'),
        os.path.join(subset_detect_dir, 'val')
    )
    # ————— WRITE subset_detect/data.yaml —————
    data_cfg = {
        'path': subset_detect_dir,   # root
        'train': 'train/images',
        'val':   'val/images',
        'test':  'test/images',
        'nc':    len(classes),
        'names': classes
    }
    with open(os.path.join(subset_detect_dir, 'data.yaml'), 'w') as f:
        yaml.safe_dump(data_cfg, f)

    print("Detection subset ready under:", subset_detect_dir)
    print("Splits:", os.listdir(subset_detect_dir))  # should list ['train','val','test']

    # 2) Train & evaluate as a detector
    model = YOLO('yolov10m.yaml', task='detect')
    run_name = f"subset_{num_samples['train']}_{num_samples['valid']}_{num_samples['test']}_det"

    model.train(
    data=os.path.join(subset_detect_dir, 'data.yaml'),
    epochs=50, imgsz=640, batch=2,
    project='bird-drone-subset', name=run_name,
    exist_ok=True, pretrained=True
    )


    results = model.val(
    data=os.path.join(subset_detect_dir, 'data.yaml'),
    plots=True
    )
    p, r, mAP50, mAP5095 = results.box.mean_results()
    print(f"Precision:      {p:.4f}")
    print(f"Recall:         {r:.4f}")
    print(f"mAP @0.50:      {mAP50:.4f}")
    print(f"mAP @0.50–0.95: {mAP5095:.4f}")
