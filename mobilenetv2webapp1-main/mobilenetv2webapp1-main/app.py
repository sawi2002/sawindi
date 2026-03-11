# app.py - Fruit Classifier Streamlit App (MobileNetV2 Transfer Learning)

import os
import json
import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model


# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered"
)

st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🍎🍌🍊 Fruit Classifier")

# ===========================
# SETTINGS (MUST MATCH TRAINING)
# ===========================
IMG_SIZE = (160, 160)
MODEL_PATH = "student_mobilenetv2_transfer_learning.keras"
CLASS_JSON = "class_names.json"

FRUIT_INFO = {
    'apple': "🍎 Rich in fiber and vitamin C. Usually comes in red and green varieties.",
    'avocado': "🥑 Packed with healthy fats and nutrients. Perfect for guacamole!",
    'banana': "🍌 Great source of potassium. Perfect pre-workout snack!",
    'cherry': "🍒 Sweet and tart! Rich in antioxidants and anti-inflammatory compounds.",
    'kiwi': "🥝 Packed with vitamin C - more than oranges!",
    'mango': "🥭 Known as the king of fruits. Rich in vitamins A and C.",
    'orange': "🍊 Famous for vitamin C. Great for boosting immunity.",
    'pineapple': "🍍 Contains bromelain, an enzyme that aids digestion.",
    'strawberries': "🍓 Loaded with vitamin C and antioxidants. Great for heart health!",
    'watermelon': "🍉 92% water! Perfect for staying hydrated in summer."
}

# ===========================
# UTIL: file hash (helps detect stale deployments)
# ===========================
def short_sha1(filepath: str) -> str:
    h = hashlib.sha1()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:10]


# ===========================
# LOAD CLASS NAMES (source of truth)
# ===========================
def load_class_names():
    if not os.path.exists(CLASS_JSON):
        st.error(f"Missing {CLASS_JSON}. Please add it to the repo.")
        st.stop()
    with open(CLASS_JSON, "r") as f:
        names = json.load(f)
    if not isinstance(names, list) or len(names) != 10:
        st.error(f"{CLASS_JSON} must be a list of 10 class names. Got: {names}")
        st.stop()
    return names

CLASS_NAMES = load_class_names()


# ===========================
# LOAD MODEL
# ===========================
# NOTE: During debugging, we avoid cache to prevent stale model.
# After everything is working, you can wrap this with @st.cache_resource.
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Missing model file: {MODEL_PATH}. Please add it to the repo.")
        st.stop()
    with st.spinner("Loading AI model..."):
        m = load_model(MODEL_PATH, compile=False)
    return m

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ===========================
# DEBUG PANEL (optional but VERY helpful)
# ===========================
with st.sidebar:
    st.header("🔎 Debug")
    st.write("TensorFlow:", tf.__version__)
    st.write("Model file exists:", os.path.exists(MODEL_PATH))
    if os.path.exists(MODEL_PATH):
        st.write("Model size (MB):", round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2))
        st.write("Model sha1:", short_sha1(MODEL_PATH))
    st.write("Model input shape:", getattr(model, "input_shape", None))
    st.write("Model output shape:", getattr(model, "output_shape", None))
    st.write("Class names:", CLASS_NAMES)

    st.markdown("---")
    st.markdown("""
**If you ever see "orange for everything":**
- It's usually **double preprocessing**
- or Streamlit running an **old cached model**
- or wrong class mapping (fixed here via `class_names.json`)
""")


# ===========================
# PREPROCESS (FOOL-PROOF)
# ===========================
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    FOOL-PROOF RULE:
    Your trained .keras model already contains MobileNetV2 preprocessing INSIDE.
    Therefore we DO NOT call preprocess_input here.

    We only:
      - resize
      - ensure RGB
      - convert to float32 in 0..255 range
      - add batch dimension
    """
    try:
        img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    except AttributeError:
        img = ImageOps.fit(img, IMG_SIZE, Image.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img).astype(np.float32)   # 0..255
    arr = np.expand_dims(arr, axis=0)          # (1,H,W,3)
    return arr


def predict_fruit(img: Image.Image):
    x = preprocess_image(img)

    # sanity check: should be near 0..255, NOT -1..1
    x_min, x_max = float(x.min()), float(x.max())

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))

    predicted_class = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100.0

    all_predictions = [
        {"fruit": CLASS_NAMES[i], "probability": float(preds[i]) * 100.0}
        for i in range(len(CLASS_NAMES))
    ]
    all_predictions.sort(key=lambda d: d["probability"], reverse=True)

    return predicted_class, confidence, all_predictions, (x_min, x_max)


# ===========================
# UI
# ===========================
st.markdown("""
Upload a fruit image and let AI identify it!  
**Supported fruits:** apple, avocado, banana, cherry, kiwi, mango, orange, pineapple, strawberries, watermelon
""")

st.divider()

uploaded_file = st.file_uploader(
    "Choose a fruit image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image of a single fruit"
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(img, use_container_width=True)

        with col2:
            st.subheader("🤖 AI Prediction")

            with st.spinner("Analyzing fruit..."):
                predicted_fruit, confidence, all_predictions, (x_min, x_max) = predict_fruit(img)

            st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h2 style='color: #2e7d32; margin: 0;'>{predicted_fruit}</h2>
                    <p style='color: #558b2f; font-size: 20px; margin: 5px 0;'>
                        Confidence: {confidence:.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence / 100.0))

            if predicted_fruit in FRUIT_INFO:
                st.info(FRUIT_INFO[predicted_fruit])

            # quick input sanity info (helps instantly spot double-preprocessing)
            st.caption(f"Input min/max fed to model: {x_min:.1f} / {x_max:.1f} (should look like 0..255-ish)")

        st.divider()
        st.subheader("📊 All Fruit Probabilities")

        df = pd.DataFrame(all_predictions)
        df["probability"] = df["probability"].map(lambda x: f"{x:.2f}%")
        df.columns = ["Fruit", "Probability"]
        df.index = range(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Probability Distribution")
        chart_data = pd.DataFrame({
            "Fruit": [p["fruit"] for p in all_predictions],
            "Probability (%)": [p["probability"] for p in all_predictions],
        })
        st.bar_chart(chart_data.set_index("Fruit"))

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.error("Please upload a valid image file.")
else:
    st.info("👆 Please upload a fruit image to get started!")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
This app uses **Transfer Learning** with **MobileNetV2** to classify fruit images.

**Tip:** If you update the model file in GitHub, use **Reboot app** in Streamlit Cloud.
""")
