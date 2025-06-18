import streamlit as st
import numpy as np
import os
import random
from PIL import Image
from utils import load_svhn_model, load_digit_struct, crop_and_resize
import pickle
import matplotlib.pyplot as plt

import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "svhn_digit_model.h5")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Z78O0_oeJ-k_onie6QZOV7YEuxjqlmH6"

print(f"Loading model from path: {MODEL_PATH}")

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            print(f"Failed to download model: Status code {response.status_code}")

download_model()

TEST_JSON_PATH = os.path.join(BASE_DIR, "..", "test", "digitStruct.json")
TEST_JSON_URL = "https://drive.google.com/uc?export=download&id=1d5IOvz-v45D7O5eJ9aK3Th_BsPxG2qnY"

def download_test_json():
    os.makedirs(os.path.dirname(TEST_JSON_PATH), exist_ok=True)
    if not os.path.exists(TEST_JSON_PATH):
        print("Downloading test digitStruct.json...")
        response = requests.get(TEST_JSON_URL)
        if response.status_code == 200:
            with open(TEST_JSON_PATH, "wb") as f:
                f.write(response.content)
            print("Test digitStruct.json downloaded successfully.")
        else:
            print(f"Failed to download test digitStruct.json: Status code {response.status_code}")

download_test_json()

# Constants
NUM_CLASSES = 11
MAX_DIGITS = 5

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "svhn_digit_model.h5")
JSON_PATH = os.path.join(BASE_DIR, "..", "test", "digitStruct.json")
HISTORY_PATH = os.path.join(BASE_DIR, "..", "model", "history.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "..", "model", "metrics.pkl")

# Load model (cached)
@st.cache_resource
def load_model_once():
    return load_svhn_model(MODEL_PATH)

model = load_model_once()

# Load history
def load_history():
    with open(HISTORY_PATH, "rb") as f:
        return pickle.load(f)

# Load metrics
def load_metrics():
    with open(METRICS_PATH, "rb") as f:
        return pickle.load(f)

# App title
st.title("📸 SVHN Digit Recognizer")
st.write("Upload an image of house numbers or select a random test image to see predictions from a deep learning model.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📷 Predict Digits", "📈 Model Stats", "ℹ️ About"])

# --------- Tab 1: Predict Digits ---------
with tab1:
    option = st.radio("Choose input type:", ["Upload Image", "Random Test Image"])
    image = None
    boxes = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a house number image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    elif option == "Random Test Image":
        metadata = load_digit_struct(JSON_PATH)
        sample = random.choice(metadata)
        image_path = os.path.join(BASE_DIR, "..", "test", sample["filename"])
        image = Image.open(image_path).convert("RGB")
        boxes = sample["boxes"]
        st.image(image, caption=f"Random Test Image: {sample['filename']}", use_container_width=True)

    if image is not None:
        if boxes:  # Random test image
            processed = crop_and_resize(image_path, boxes)
        else:  # Uploaded image
            processed = image.resize((64, 64))
            processed = np.array(processed) / 255.0

        input_tensor = np.expand_dims(processed, axis=0)
        predictions = model.predict(input_tensor)
        pred_digits = np.argmax(predictions, axis=-1)[0]
        pred_cleaned = [str(d) if d != 10 else "-" for d in pred_digits]

        st.subheader("🔢 Predicted Digit Sequence:")
        st.write(" ".join(pred_cleaned))

# --------- Tab 2: Model Stats ---------
with tab2:
    st.subheader("📈 Model Statistics")

    try:
        history = load_history()
        metrics = load_metrics()

        # Sequence accuracy metric
        st.metric("✅ Sequence Accuracy", f"{metrics['sequence_accuracy']:.2%}")

        # Digit level accuracy
        st.write("📊 **Per-Digit Accuracy**")
        for i, acc in enumerate(metrics["digit_accuracies"], start=1):
            st.write(f"Digit {i}: {acc:.2%}")

        # Training curves
        st.write("📈 **Training Curves**")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Val Accuracy')
        ax1.set_title("Accuracy Over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Val Loss')
        ax2.set_title("Loss Over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading model stats: {e}")

# --------- Tab 3: About ---------
with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    This app uses a Convolutional Neural Network (CNN) to detect multi-digit numbers from images in the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/).  
    It was developed as part of a deep learning course project and fine-tuned for digit-level accuracy.  
    **Author:** Jake Bronkala  
    **Model Accuracy:** Final version achieved over 84% accuracy.

    During training, the loss was still decreasing and accuracy improving even after the 25 epochs we used, indicating the model could likely improve further.  
    With more computing power and extended training time, even higher accuracy results are achievable.
    """)