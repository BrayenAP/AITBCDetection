import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse
import os

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using image :", args.image)
print("Using model :", args.model)
print("Saving to   :", args.output)

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# LOAD MODEL
base_model = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool='avg')

model = nn.Sequential(
    base_model,
    nn.Linear(768, 2)
)

model.load_state_dict(torch.load(args.model, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# TARGET LAYER
target_layer = model[0].stages[2].blocks[-1]

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# LOAD IMAGE
img = Image.open(args.image).convert("RGB")
orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# FORWARD & BACKWARD
output = model(input_tensor)
class_idx = output.argmax(dim=1).item()

model.zero_grad()
output[0, class_idx].backward()

# BUILD CAM
grads = gradients.detach().cpu().numpy()[0]      # (C,H,W)
acts = activations.detach().cpu().numpy()[0]     # (C,H,W)

weights = grads.mean(axis=(1, 2))                # GAP
cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)

# safe normalize
cam = cam / (cam.max() + 1e-8)

# RESIZE
cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))

# LUNG PRIOR MASK 
h, w, _ = orig.shape
mask = np.zeros_like(cam)

# Koordinat Box
mask[int(0.18*h):int(0.88*h), int(0.15*w):int(0.85*w)] = 1
mask = cv2.GaussianBlur(mask, (51, 51), 0) # Soft edges

cam = cam * mask

# OVERLAY
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

# SAVE
os.makedirs(os.path.dirname(args.output), exist_ok=True)
cv2.imwrite(args.output, overlay)

print("Grad-CAM lung-focused saved successfully.")