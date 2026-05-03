import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")

# =========================
# CLASS LABELS
# =========================
label_map = {
    0: "Melanoma",
    1: "Nevus",
    2: "Basal Cell Carcinoma",
    3: "Benign Keratosis",
    4: "Dermatofibroma",
    5: "Vascular Lesion",
    6: "Actinic Keratosis"
}

num_classes = len(label_map)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL (FROM GOOGLE DRIVE)
# =========================
@st.cache_resource
def load_model():
    model_path = "skin_cancer_model.pth"

    # 🔽 Google Drive direct download link
    url = "https://drive.google.com/uc?id=1GDD2iC0qD7Md_srGr5HXqUDQpu6de0-X"

    # 🔽 Download model if not already present
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... please wait ⏳"):
            gdown.download(url, model_path, quiet=False)

    # 🔽 Load model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

model = load_model()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# UI
# =========================
st.title("🧠 Skin Cancer Detection System")
st.write("Upload a dermoscopic image to predict skin condition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = label_map[predicted.item()]
    confidence_score = confidence.item() * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    # Optional breakdown
    with st.expander("See all class probabilities"):
        for i in range(num_classes):
            st.write(f"{label_map[i]}: {probabilities[0][i].item()*100:.2f}%")
