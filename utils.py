print("#TEST utils.py loaded")


import json
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_URL = "https://drive.google.com/uc?export=download&id=1Z78O0_oeJ-k_onie6QZOV7YEuxjqlmH6"

def load_svhn_model(path):
    full_path = os.path.abspath(path)
    model_dir = os.path.dirname(full_path)

    if not os.path.exists(full_path):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        print(f"Model file not found at {full_path}. Downloading now...")
        gdown.download(MODEL_URL, full_path, quiet=False)
        
        # Check if file now exists after download
        if os.path.exists(full_path):
            print(f"Model successfully downloaded to {full_path}")
        else:
            raise FileNotFoundError(f"Failed to download the model to {full_path}")
    else:
        print(f"Model file already exists at {full_path}")

    return load_model(full_path)



def load_digit_struct(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for entry in data:
        filename = entry["filename"]
        boxes = sorted(entry["boxes"], key=lambda b: b["left"])
        results.append({
            "filename": filename,
            "digits": [int(b["label"]) if int(b["label"]) != 10 else 0 for b in boxes],
            "boxes": boxes
        })
    return results

def crop_and_resize(image_path, boxes, size=(64, 64)):
    img = Image.open(image_path).convert("RGB")
    left = min([b["left"] for b in boxes])
    top = min([b["top"] for b in boxes])
    right = max([b["left"] + b["width"] for b in boxes])
    bottom = max([b["top"] + b["height"] for b in boxes])
    cropped = img.crop((left, top, right, bottom))
    resized = cropped.resize(size)
    return np.array(resized) / 255.0

def load_svhn_model(path="model/svhn_digit_model.h5"):
    import os
    full_path = os.path.join(os.path.dirname(__file__), '..', path)
    print("Trying to load model from:", full_path)
    return load_model(full_path)

import pickle

def load_training_metrics(history_path, metrics_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    return history, metrics
