import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import subprocess
import os
import uuid
import sys

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_A_PATH = os.path.join(BASE_DIR, "model", "model_A_weighted.pth")
MODEL_B_PATH = os.path.join(BASE_DIR, "models", "model_B_best.pth")

GRADCAM_FULL = os.path.join(BASE_DIR, "scripts", "gradcam.py")
GRADCAM_LUNG = os.path.join(BASE_DIR, "scripts", "gradcam_paru.py")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PAGE SETUP
st.set_page_config(
    page_title="TB Detection System",
    layout="centered"
)

st.title("🫁 Tuberculosis Detection from Chest X-ray")
st.caption("ConvNeXt-based Explainable AI System")

# MODEL LOADER
@st.cache_resource
def load_model(model_path):
    backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
    model = nn.Sequential(
        backbone,
        nn.Linear(768, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# UI INPUTS
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

model_choice = st.selectbox(
    "Select Detection Model",
    [
        "Model A – Weighted (Real-world distribution)",
        "Model B – Balanced (Equal class distribution)"
    ]
)

cam_choice = st.selectbox(
    "Select Explanation Method",
    [
        "Grad-CAM Full Image",
        "Grad-CAM Lung Focused"
    ]
)

analyze = st.button("🔍 Analyze")

# MAIN LOGIC
if uploaded_file and analyze:

    # SAVE UPLOADED IMAGE
    image_id = str(uuid.uuid4())[:8]
    image_path = os.path.join(UPLOAD_DIR, f"{image_id}_{uploaded_file.name}")

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # MODEL SELECTION
    if "Model A" in model_choice:
        model_path = MODEL_A_PATH
        model_name = "Model A (Weighted)"
    else:
        model_path = MODEL_B_PATH
        model_name = "Model B (Balanced)"

    model = load_model(model_path)

    # INFERENCE
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "Tuberculosis" if pred.item() == 1 else "Normal"

    st.subheader("🧠 Prediction Result")
    st.write(f"**Model Used:** {model_name}")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {conf.item() * 100:.2f}%")

    # GRAD-CAM
    st.subheader("🖼️ Model Explanation (Grad-CAM)")

    output_cam_path = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{image_id}.jpg"
    )

    gradcam_script = GRADCAM_FULL if "Full" in cam_choice else GRADCAM_LUNG

    result = subprocess.run(
        [
            sys.executable,
            gradcam_script,
            "--image", image_path,
            "--model", model_path,
            "--output", output_cam_path
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        st.error("Grad-CAM script failed")
        st.text(result.stderr)
    else:
        if os.path.exists(output_cam_path):
            st.image(
                output_cam_path,
                caption=f"{cam_choice} ({model_name})",
                use_container_width=True
            )
        else:
            st.error("Grad-CAM did not generate output.")