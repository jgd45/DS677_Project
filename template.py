#Download DATASET

import os
import shutil
from kagglehub import dataset_download
from glob import glob
import matplotlib.pyplot as plt

# download dataset
dataset = "stealthknight/bird-vs-drone"
path = dataset_download(dataset)
print("Root directory:", path)
print("Contents:", os.listdir(path))

# Define dataset base
base_dir = os.path.join(path, "Dataset")


#RELABEL DATASET SO IT MATACHES 0 FOR BIRD AND 1 FOR DRONE

import os
import shutil


src = "/kaggle/input/bird-vs-drone"
dst = "/kaggle/working/Dataset/Dataset_editable"
if not os.path.exists(dst):
    print(f"Copying dataset from {src} to {dst} …")
    shutil.copytree(src, dst)
else:
    print(f"Writable copy already exists at {dst}")


base_dir = dst


prefix_class_map = {
    'BTR': 0, 'BV': 0, 'BT': 0,   # bird
    'DTR': 1, 'DV': 1, 'DT': 1    # drone
}

def fix_and_check_labels(labels_dir):

    total_files = 0
    corrected_files = 0


    prefixes = sorted(prefix_class_map.keys(), key=lambda x: -len(x))

    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue

        total_files += 1
        expected = None
        for pref in prefixes:
            if fname.startswith(pref):
                expected = prefix_class_map[pref]
                break
        if expected is None:

            continue

        path = os.path.join(labels_dir, fname)
        lines = open(path, 'r').read().splitlines()
        new_lines = []
        need_rewrite = False

        for line in lines:
            parts = line.split()
            actual = float(parts[0])
            if actual != expected:
                parts[0] = str(expected)
                need_rewrite = True
            new_lines.append(' '.join(parts))

        if need_rewrite:
            with open(path, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
            corrected_files += 1

    print(f"Checked {total_files} files in {os.path.basename(labels_dir)}; Corrected {corrected_files}")


for split in ["train", "valid", "test"]:
    lbl_dir = os.path.join(base_dir, "Dataset", split, "labels")
    fix_and_check_labels(lbl_dir)
    
    
    
#SOME FILES HAVE SEGMENTATION SO THIS CODE CONVERTS DATASET TO BOUNDING BOXES SO THAT WE CAN RUN IT THROUGH YOLOV10

from pathlib import Path

base_dir = Path(base_dir)
output_base = base_dir.parent / (base_dir.name + "-converted")
output_base.mkdir(parents=True, exist_ok=True)

for split in ("train", "valid", "test"):
    img_src = base_dir / split / "images"
    lbl_src = base_dir / split / "labels"
    img_dst = output_base / split / "images"
    lbl_dst = output_base / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img_file in img_src.glob("*.*"):
        dst_img = img_dst / img_file.name
        if not dst_img.exists():
            dst_img.write_bytes(img_file.read_bytes())
    for lbl_file in lbl_src.glob("*.txt"):
        tokens = list(map(float, lbl_file.read_text().split()))
        i = 0
        new_lines = []
        while i < len(tokens):
            cls_id = int(tokens[i])
            i += 1
            points = []
            while i+1 < len(tokens) and not (tokens[i] in [0,1] and (0 <= tokens[i+1] <= 1)):
                points.append((tokens[i], tokens[i+1]))
                i += 2
            if len(points) >= 2:
                xs = [x for x, _ in points]
                ys = [y for _, y in points]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                width = max_x - min_x
                height = max_y - min_y
                new_lines.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        dst_lbl = lbl_dst / lbl_file.name
        dst_lbl.write_text("\n".join(new_lines))

print("Dataset conversion complete!")



#SETTING UP YAML FOR SPECIFIC SETUP


output_base = str(output_base)

with open('data.yaml', 'w') as f:
    f.write(f"""
path: {output_base}
train: train/images
val: valid/images
nc: 2
names: ['bird', 'drone']
""".strip())

print("data.yaml created!")


#!cat data.yaml





#FIX CLASS IMBALANCE ADD MORE BIRD IMAGES THROUGH PIPELINE 

from glob import glob

def count_labels(label_dir):
    bird, drone = 0, 0
    for label_file in glob(os.path.join(label_dir, "*.txt")):
        with open(label_file, 'r') as f:
            for line in f:
                cls = int(line.split()[0])
                if cls == 0:
                    bird += 1
                elif cls == 1:
                    drone += 1
    return bird, drone

train_lbl_dir = os.path.join(base_dir, "train", "labels")
bird_count, drone_count = count_labels(train_lbl_dir)

print(f"Bird labels:  {bird_count}")
print(f"Drone labels: {drone_count}")

#AUGEMENTATION PIPELINE TO ADD TO BIRD CLASS

import os
import cv2
import numpy as np
import random
from glob import glob
import albumentations as A





train_img_dir = os.path.join(base_dir, "train", "images")
train_lbl_dir = os.path.join(base_dir, "train", "labels")


augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.2),
    A.Rotate(limit=10, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.3),
],
bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))


bird_imgs = []
for lbl_file in glob(os.path.join(train_lbl_dir, "*.txt")):
    with open(lbl_file, 'r') as f:
        if any(line.startswith("0 ") for line in f):
            bird_imgs.append(os.path.basename(lbl_file).replace(".txt", ".jpg"))


random.shuffle(bird_imgs)
aug_needed = 11500
repeat = 2
subset = bird_imgs[: aug_needed // repeat]


count = 0
for fname in subset:
    img_path = os.path.join(train_img_dir, fname)
    lbl_path = os.path.join(train_lbl_dir, fname.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)
    if image is None:
        continue

    # load original boxes
    bboxes, class_labels = [], []
    for line in open(lbl_path, 'r'):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = float(parts[0])
        box = list(map(float, parts[1:]))
        # initial clamp
        box = [min(max(v, 0.0), 1.0) for v in box]
        bboxes.append(box)
        class_labels.append(cls)
    if not bboxes:
        continue

    # apply augmentations
    for i in range(repeat):
        try:
            aug = augment(image=image, bboxes=bboxes, class_labels=class_labels)
        except ValueError:
            # skip if any box went out of bounds
            continue

        aug_image = aug['image']
        filtered_bboxes, filtered_labels = [], []
        for cls, box in zip(aug['class_labels'], aug['bboxes']):
            # clamp coords back into [0,1]
            xc, yc, bw, bh = [max(min(v, 1.0), 0.0) for v in box]
            # skip degenerate boxes
            if bw <= 0 or bh <= 0:
                continue
            filtered_bboxes.append([xc, yc, bw, bh])
            filtered_labels.append(cls)

        if not filtered_bboxes:
            continue

        # save augmented image + labels
        new_base = fname.replace(".jpg", f"_aug_{i}")
        out_img = os.path.join(train_img_dir, new_base + ".jpg")
        out_lbl = os.path.join(train_lbl_dir, new_base + ".txt")

        cv2.imwrite(out_img, aug_image)
        with open(out_lbl, 'w') as f:
            for cls, box in zip(filtered_labels, filtered_bboxes):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")

        count += 1

print(f"Created {count} new augmented bird samples.")






















# ------------------------------ THIS CODE IS FOR HYPER PARAMETER TUNING --------------
#IF NEEDED WE ARE USING OPTUNA TO HYPER PARAMETER FIT


#!pip install optuna

#WE ARE USING YOLO (S) TO TRAIN HYPERPARAMTERS INSTEAD OF B , BECAUSE OF FASTER TRAINING

#!wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt


#SETTING UP TRAINING MODEL METHOD WITH OPTIMAL RUN CONFIGS 
#----------- DO NOT RE REUN CELL ------------------------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
os.makedirs("/content/drive/MyDrive/yolo_optuna", exist_ok=True)

STUDY_DB_PATH = "/content/drive/MyDrive/yolo_optuna/yolov10_optuna.db"


import optuna
import random
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
random.seed(42)


def train_model(config, epochs=10, patience=3):
    model = YOLO("/content/yolov10/yolov10s.pt")
    result = model.train(
        data="data.yaml",
        imgsz=config["imgsz"],
        batch=config["batch"],
        epochs=epochs,
        patience=patience,
        warmup_epochs=1.0,
        cos_lr=True,
        deterministic=True,
        device=0,
        val=True,
        name=f"optuna_run_{config['trial_id']}",
        exist_ok=True,
        lr0=config["lr0"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        workers=os.cpu_count(),
        cache=False,
    )
    if isinstance(result, DetMetrics):
        return result.box.map50
    else:
        raise TypeError(f"Unexpected result type: {type(result)}")
    
#RUN HYPER PARAMETER TUNING CODE

def objective(trial):
    config = {
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
        "momentum": trial.suggest_float("momentum", 0.85, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", [8, 16]),
        "imgsz": trial.suggest_categorical("imgsz", [416, 640]),
        "trial_id": trial.number,
    }
    try:
        return train_model(config, epochs=10)
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

storage_url = f"sqlite:///{STUDY_DB_PATH}"
study = optuna.create_study(
    direction="maximize",
    study_name="yolov10_optuna_study",
    storage=storage_url,
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
)

study.optimize(objective, n_trials=20)

print("Best hyperparameters found:")
print(study.best_params)
print("Best mAP50:", study.best_value)

