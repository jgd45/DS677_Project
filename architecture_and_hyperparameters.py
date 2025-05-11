#Download DATASET

from __future__ import annotations

import os, random, shutil, sys, yaml
from pathlib import Path

# --------------------------------------------------------------------------- #
# Always use *your* local Ultralytics fork
ULTRA_PATH = "/home/ordan_avis/DS677_Project/ultralytics"
sys.path.insert(0, ULTRA_PATH)          # ← highest priority in sys.path

from ultralytics import YOLO            # now guaranteed to come from ULTRA_PATH
# --------------------------------------------------------------------------- #

# ----------------------------- CONFIG -------------------------------------- #
BASE          = Path("/home/ordan_avis/DS677/Dataset")  # full dataset root
SPLITS        = ["train", "valid", "test"]
NUM_SAMPLES   = {"train": 2000, "valid": 500, "test": 500}
CLASSES       = ["bird", "drone"]
PREFIX_MAP    = {
    "train": ("BT", "DT"),
    "valid": ("BV", "DV"),
    "test":  ("BT", "DT"),
}
TASK          = "detect"          # "classify" or "detect"
# --------------------------------------------------------------------------- #
def safe_run_name(base_dir: Path, run_name: str) -> str:
    """Prevent overwriting by incrementing run name if folder exists."""
    path = base_dir / run_name
    count = 1
    while path.exists():
        path = base_dir / f"{run_name}_v{count}"
        count += 1
    return path.name

def make_subset_classify() -> Path:
    """Build /subset/ in classification folder hierarchy."""
    subset = BASE / "subset"
    if subset.exists():
        shutil.rmtree(subset)
    for split in SPLITS:
        bird_pref, drone_pref = PREFIX_MAP[split]
        src_dir = BASE / split / "images"
        dst_split = subset / split / "images"
        dst_split.mkdir(parents=True, exist_ok=True)

        imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".png"))]
        bird = [f for f in imgs if f.startswith(bird_pref)]
        drone = [f for f in imgs if f.startswith(drone_pref)]
        k = NUM_SAMPLES[split] // 2
        chosen = random.sample(bird, min(k, len(bird))) + \
                 random.sample(drone, min(k, len(drone)))
        while len(chosen) < NUM_SAMPLES[split]:
            chosen.append(random.choice(imgs))

        for fn in chosen:
            shutil.copy(src_dir / fn, dst_split / fn)

        # reorganise into {bird,drone} folders
        for cls in CLASSES:
            (subset / split / cls).mkdir(parents=True, exist_ok=True)

        for fn in os.listdir(dst_split):
            dst_cls = "bird" if fn.startswith(bird_pref) else "drone"
            shutil.move(dst_split / fn, subset / split / dst_cls / fn)
        shutil.rmtree(dst_split)

    # rename valid→val
    (subset / "valid").rename(subset / "val")
    print("Classification subset ready:", subset)
    return subset


def make_subset_detect() -> tuple[Path, Path]:
    """Build /subset_detect/ and its data.yaml for detection."""
    subset = BASE / "subset_detect"
    if subset.exists():
        shutil.rmtree(subset)
    for split in SPLITS:
        (subset / split / "images").mkdir(parents=True, exist_ok=True)
        (subset / split / "labels").mkdir(parents=True, exist_ok=True)

        src_imgs = BASE / split / "images"
        src_lbls = BASE / split / "labels"
        imgs = [f for f in os.listdir(src_imgs) if f.lower().endswith((".jpg", ".png"))]
        chosen = random.sample(imgs, min(NUM_SAMPLES[split], len(imgs)))
        for fn in chosen:
            shutil.copy(src_imgs / fn, subset / split / "images" / fn)
            lbl = src_lbls / f"{Path(fn).stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, subset / split / "labels" / lbl.name)

    (subset / "valid").rename(subset / "val")

    data_yaml = subset / "data.yaml"
    with data_yaml.open("w") as f:
        yaml.safe_dump(
            {
                "path": str(subset),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": len(CLASSES),
                "names": CLASSES,
            },
            f,
        )
    print("Detection subset ready:", subset)
    return subset, data_yaml


def train_classify(root: Path):
    model = YOLO("yolov10b.yaml", task="classify")
    base_name = f"subset_{NUM_SAMPLES['train']}_{NUM_SAMPLES['valid']}_{NUM_SAMPLES['test']}"
    run_name = safe_run_name(Path("bird-drone-subset"), base_name)
    model.train(
    data=str(root),
    epochs=100,
    imgsz=416,
    batch=2,
    lr0=0.0001690947468168212,
    momentum=0.8573102587296049,
    weight_decay=1.3724756935926911e-05,
    project='bird-drone-subset',
    name=run_name,
    pretrained=True
    )
    metrics = model.val(data=str(root), plots=True)
    print(f"Top‑1 Acc {metrics.top1:.4f} | Top‑5 Acc {metrics.top5:.4f}")


def train_detect(root: Path, data_yaml: Path):
    model = YOLO("yolov10m.yaml", task="detect")
    base_name = f"subset_{NUM_SAMPLES['train']}_{NUM_SAMPLES['valid']}_{NUM_SAMPLES['test']}_det"
    run_name = safe_run_name(Path("bird-drone-subset"), base_name)
    
    model.train(
    data=str(data_yaml),  
    epochs=100,
    imgsz=416,  
    batch=2,     
    lr0=0.0001690947468168212,
    momentum=0.8573102587296049,
    weight_decay=1.3724756935926911e-05,
    project='bird-drone-subset',
    name=run_name,
    exist_ok=True,
    pretrained=True
    )
    
    res = model.val(data=str(data_yaml), plots=True)
    p, r, m50, m5095 = res.box.mean_results()
    print(f"P {p:.4f} | R {r:.4f} | mAP50 {m50:.4f} | mAP50‑95 {m5095:.4f}")


if __name__ == "__main__":
    random.seed(0)

    if TASK == "classify":
        subset_root = make_subset_classify()
        train_classify(subset_root)
    else:
        subset_root, yaml_file = make_subset_detect()
        train_detect(subset_root, yaml_file)
