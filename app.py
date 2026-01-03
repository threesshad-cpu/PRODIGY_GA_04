import os
import streamlit as st
import tensorflow as tf
import numpy as np
import io
from PIL import Image, ImageEnhance

# Suppress technical logs for a clean terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Compact Layout & Moving Aurora CSS ---
st.set_page_config(page_title="AI Facade Architect Pro", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #0f172a);
        background-size: 400% 400%;
        animation: aurora_flow 5s ease infinite;
        color: #f8fafc;
    }

    @keyframes aurora_flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Compact Typography */
    .elite-header {
        text-align: center;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2rem !important;
        margin: 0;
        padding: 0;
    }

    /* Fixed Image Size Constraints */
    [data-testid="stImage"] img {
        max-width: 320px !important;
        max-height: 320px !important;
        border-radius: 8px;
        object-fit: contain;
    }

    /* Action Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        border: none !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Predict & Load Logic ---
@st.cache_resource
def load_cmp_model():
    model_path = 'gen.h5'
    if not os.path.exists(model_path):
        return None
    try:
        # safe_mode=False avoids 'quantization_mode' attribute errors in Keras 3.0
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except:
        return None

def process_img(image):
    # Standard Pix2Pix scaling to 256x256
    image = image.resize((256, 256))
    img_array = (np.array(image).astype(np.float32) / 127.5) - 1
    return np.expand_dims(img_array, axis=0)

def enhance_result(prediction):
    # Denormalize back to [0, 255]
    out = (prediction[0] + 1) * 127.5
    img = Image.fromarray(out.astype(np.uint8))
    # Production sharpening boost
    return ImageEnhance.Sharpness(img).enhance(1.6)

# --- 3. Main Application ---
st.markdown('<div class="elite-header">AI FACADE ARCHITECT</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-size: 0.85rem; margin-bottom: 15px;'>Task-04: Neural Image-to-Image Synthesis</p>", unsafe_allow_html=True)

# Centered Uploader
_, upload_col, _ = st.columns([1, 2, 1])
with upload_col:
    file = st.file_uploader("Upload Architectural Map", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if file:
    input_img = Image.open(file).convert('RGB')
    
    # Using 4 columns [1, 2, 2, 1] creates a compact, centered display
    _, col1, col2, _ = st.columns([0.8, 2, 2, 0.8])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("**Source Labels**")
        # width='stretch' replaces use_container_width=True for 2026 API
        st.image(input_img, width='stretch') 
        st.markdown('</div>', unsafe_allow_html=True)
        
        execute = st.button("🚀 EXECUTE")

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("**Neural Output**")
        
        if execute:
            with st.spinner("Synthesizing..."):
                model = load_cmp_model()
                if model:
                    pred = model.predict(process_img(input_img), verbose=0)
                    result = enhance_result(pred)
                    st.image(result, width='stretch')
                    
                    # Minimal Export
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button("📥 SAVE", buf.getvalue(), "render.png")
        else:
            st.info("Ready for synthesis")
        st.markdown('</div>', unsafe_allow_html=True)
