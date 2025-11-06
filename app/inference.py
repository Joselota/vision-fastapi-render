# app/inference.py
from __future__ import annotations
import os, io, json
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    from keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/modelo_insectos.h5"))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", "models/classes.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

_model: tf.keras.Model | None = None
_classes: Dict[int, str] | None = None

def load_artifacts():
    global _model, _classes
    if _model is None:
        _model = load_model(str(MODEL_PATH), compile=False)

    if _classes is None:
        data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        _classes = {int(k): v for k, v in data.items()}
    return _model, _classes

def prepare_image(file_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32)[None, ...]
    # Si en tu training usaste SOLO rescale=1./255, reemplaza por: x = x / 255.0
    x = preprocess_input(x)
    return x

def predict_bytes(file_bytes: bytes) -> dict:
    model, classes = load_artifacts()
    x = prepare_image(file_bytes)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    if len(classes) == len(preds):
        proba = {classes[i]: float(round(preds[i], 6)) for i in range(len(preds))}
        label = classes[idx]
    else:
        proba = {str(i): float(round(preds[i], 6)) for i in range(len(preds))}
        label = str(idx)
    return {"label": label, "proba": proba, "model_version": MODEL_VERSION}
