import streamlit as st
import pydicom
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Parkinson Prediction System", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Times New Roman', serif;
        }
        .main {
            background-color: #121212;
        }
        .big-title {
            font-size: 80px;
            font-weight: bold;
            text-align: center;
            color: #ffcc00;
            text-shadow: 4px 4px 7px rgba(255,255,255,0.4);
        }
        .sub-header {
            font-size: 30px;
            font-weight: bold;
            color: #00ffaa;
        }
        .regular-text {
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# Hide sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNavToggle"] {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

if st.button("🏠 Home"):
     st.switch_page("Landing_page.py")

st.markdown('<p class="big-title">🧠 Parkinson Prediction System</p>', unsafe_allow_html=True)

# -------------------------------
# Load the Pretrained Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "C:/Users/Shraddha/PD PROJECT/efficientnetb0_model.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# DICOM Image Preprocessing
# -------------------------------
def preprocess_dicom(dicom_data):
    """Preprocess the uploaded DICOM image."""
    try:
        image = dicom_data.pixel_array.astype(np.float32)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"❌ Error processing DICOM file: {e}")
        return None

# -------------------------------
# File Uploader for DICOM Files
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a DICOM File (.dcm)", type=["dcm"])

if uploaded_file is not None:
    try:
        dicom_data = pydicom.dcmread(BytesIO(uploaded_file.getvalue()))
        image_data = dicom_data.pixel_array
        height, width = image_data.shape
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 8  # Small Circle

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<p class="sub-header">🖼️ DICOM Image</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(image_data, cmap="gray")
            circle = plt.Circle((center_x, center_y), radius, color="red", linewidth=1, fill=False)
            ax.add_patch(circle)
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown('<p class="sub-header">📋 DICOM Metadata</p>', unsafe_allow_html=True)
            metadata_keys = ["PatientName", "PatientID", "StudyDate", "Modality"]
            for key in metadata_keys:
                if hasattr(dicom_data, key):
                    st.markdown(f'<p class="regular-text"><b>{key}:</b> {getattr(dicom_data, key)}</p>', unsafe_allow_html=True)

        # -------------------------------
        # Predict Button
        # -------------------------------
        if st.button("🧠 Predict Parkinson's Disease"):
            if model:
                image_input = preprocess_dicom(dicom_data)
                if image_input is not None:
                    predictions = model.predict(image_input)
                    class_idx = np.argmax(predictions, axis=1)[0]
                    label_dict_inv = {0: '✅ CONTROL', 1: '⚠️ Parkinson\'s Detected', 2: '⚠️ Prodromal Stage'}
                    prediction_result = label_dict_inv[class_idx]

                    if class_idx == 0:
                        st.success(prediction_result)
                    else:
                        st.error(prediction_result)
            else:
                st.error("❌ Model not loaded properly. Please check the model path and retry.")

    except Exception as e:
        st.error(f"❌ Error reading DICOM file: {e}")
