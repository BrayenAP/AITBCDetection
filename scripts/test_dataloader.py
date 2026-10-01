import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "validation")

# Same transforms as training later
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=test_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Test: get 1 batch from train
images, labels = next(iter(train_loader))

print("Images batch shape: ", images.shape)
print("Labels batch shape: ", labels.shape)

print("Example labels: ", labels[:16])  # print 16 labels
print("Class to index mapping: ", train_dataset.class_to_idx)