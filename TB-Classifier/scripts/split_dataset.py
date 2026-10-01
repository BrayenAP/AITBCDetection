import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "validation")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

CLASSES = ["Normal", "Tuberculosis"]

for cls in CLASSES:
    print(f"Processing {cls}...")

    src_folder = os.path.join(RAW_DIR, cls)
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]

    # split 80% train, 20% validation
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # make class folders
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    # copy train images
    for img in train_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    # copy validation images
    for img in val_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(VAL_DIR, cls, img)
        )

print("\nDataset split completed!")