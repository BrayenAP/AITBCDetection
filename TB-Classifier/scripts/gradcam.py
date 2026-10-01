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

IMAGE_PATH = args.image
MODEL_PATH = args.model
OUTPUT_PATH = args.output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
base = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool='avg')
model = nn.Sequential(
    base,
    nn.Linear(768, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# TARGET LAYER FIX
target_layer = model[0].stages[3].blocks[-1] 

activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# LOAD IMAGE
img = Image.open(IMAGE_PATH).convert("RGB")
orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 

input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# FORWARD + BACKWARD
output = model(input_tensor)
class_idx = output.argmax(dim=1).item() # Prediksi kelas tertinggi (0 atau 1)

model.zero_grad()
output[0, class_idx].backward()

# BUILD CAM
grads = gradients.detach().cpu().numpy()[0]
acts = activations.detach().cpu().numpy()[0]

weights = grads.mean(axis=(1, 2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
# Avoid division by zero
cam /= (cam.max() + 1e-8)

# RESIZE & OVERLAY
cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# Overlay Logic
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, overlay)

print("Grad-CAM saved to:", OUTPUT_PATH)