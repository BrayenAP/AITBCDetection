import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import os

# CONFIG
MODEL_TYPE = "B"   # "A" or "B"

BASE_ROOT = os.path.dirname(os.path.abspath(__file__))

if MODEL_TYPE == "A":
    DATASET_DIR = os.path.join(BASE_ROOT, "dataset")
    MODEL_PATH  = os.path.join(BASE_ROOT, "model", "model_A_weighted.pth")
else:
    DATASET_DIR = os.path.join(BASE_ROOT, "dataset_balanced")
    MODEL_PATH  = os.path.join(BASE_ROOT, "models", "model_B_best.pth")

VAL_DIR = os.path.join(DATASET_DIR, "validation")
BATCH_SIZE = 32

# Transforms
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset & Loader
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = val_dataset.classes  # ['Normal', 'Tuberculosis']

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=0
)

model = nn.Sequential(
    backbone,
    nn.Linear(768, 2)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Inference
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1]  # TB probability
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

# Metrics
acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="binary")
rec  = recall_score(all_labels, all_preds, average="binary")   # Sensitivity
f1   = f1_score(all_labels, all_preds, average="binary")

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("===== VALIDATION METRICS =====")
print(f"Model Type: {MODEL_TYPE}")
print(f"Accuracy:     {acc*100:.2f}%")
print(f"Precision:    {prec*100:.2f}%")
print(f"Recall:       {rec*100:.2f}% (Sensitivity)")
print(f"Specificity:  {specificity*100:.2f}%")
print(f"F1-score:     {f1*100:.2f}%")

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Model {MODEL_TYPE})")
plt.tight_layout()
plt.show()

# ROC Curve & AUC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC Curve (Model {MODEL_TYPE})")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()