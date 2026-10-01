import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
import timm
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "validation")

SAVE_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(SAVE_DIR, exist_ok=True)

save_path = os.path.join(SAVE_DIR, "model_A_weighted.pth")


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Dataset
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

# Balanced Class Weights
train_counts = Counter(train_dataset.targets)
print("Sample count per class:", train_counts)

normal_count = train_counts[0]
tb_count     = train_counts[1]
total_samples = normal_count + tb_count
num_classes = 2

weight_normal = total_samples / (num_classes * normal_count)
weight_tb     = total_samples / (num_classes * tb_count)

class_weights = torch.tensor([weight_normal, weight_tb], dtype=torch.float)
print("Balanced class weights:", class_weights)

# ConvNeXt Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)

model = nn.Sequential(
    base_model,
    nn.Linear(768, 2)   # 768 → feature dim, 2 → classes
).to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training Loop + Early Stopping
num_epochs = 20
patience = 2           # stop if val accuracy doesn't improve for 2 epochs
patience_counter = 0
best_acc = 0.0

for epoch in range(num_epochs):

    # TRAIN 
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # VALIDATION
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}  — Loss: {avg_loss:.4f} — Val Acc: {val_acc:.2f}%")

    # EARLY STOPPING CHECK
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0  # reset counter
        torch.save(model.state_dict(), save_path)
        print(f"✔ New best model saved! Val Acc = {best_acc:.2f}%")

    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break

print(f"\nTraining complete. Best Val Acc = {best_acc:.2f}%")
print(f"Best model saved at: {save_path}")