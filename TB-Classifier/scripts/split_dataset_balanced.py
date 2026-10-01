import os
import random
import shutil
from sklearn.model_selection import train_test_split

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
OUT_DIR = os.path.join(BASE_DIR, "dataset_balanced")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASSES = ["Normal", "Tuberculosis"]

random.seed(RANDOM_SEED)

# Create output folders
for split in ["train", "validation"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

# Load file paths
files = {}

for cls in CLASSES:
    cls_dir = os.path.join(RAW_DIR, cls)
    files[cls] = [
        os.path.join(cls_dir, f)
        for f in os.listdir(cls_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

# BALANCING (1:1)
min_count = min(len(files["Normal"]), len(files["Tuberculosis"]))

files["Normal"] = random.sample(files["Normal"], min_count)
files["Tuberculosis"] = random.sample(files["Tuberculosis"], min_count)

print(f"Balanced dataset size per class: {min_count}")

# Train / Validation split
for cls in CLASSES:
    train_files, val_files = train_test_split(
        files[cls],
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    for f in train_files:
        shutil.copy(
            f,
            os.path.join(OUT_DIR, "train", cls, os.path.basename(f))
        )

    for f in val_files:
        shutil.copy(
            f,
            os.path.join(OUT_DIR, "validation", cls, os.path.basename(f))
        )

print("✅ Balanced dataset split completed!")