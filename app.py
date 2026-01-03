import os
import streamlit as st
import tensorflow as tf
import numpy as np
import io
from PIL import Image, ImageEnhance

# Suppress technical logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Compact Aurora UI Styling ---
st.set_page_config(page_title="🏢AI Facade Architect Pro", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #0f172a);
        background-size: 400% 400%;
        animation: aurora_flow 5s ease infinite;
    }
    @keyframes aurora_flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stImage > img {
        max-width: 320px !important;
        max-height: 320px !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Model Logic ---
@st.cache_resource
def load_cmp_model():
    model_path = 'gen.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    return None

def predict_facade(image, model):
    # Preprocess
    img_res = image.resize((256, 256))
    img_arr = (np.array(img_res).astype(np.float32) / 127.5) - 1
    input_tensor = np.expand_dims(img_arr, axis=0)
    
    # Predict
    pred = model.predict(input_tensor, verbose=0)
    
    # Post-process
    out = (pred[0] + 1) * 127.5
    res_img = Image.fromarray(out.astype(np.uint8))
    return ImageEnhance.Sharpness(res_img).enhance(1.6)

# --- 3. Sidebar: Sample Management ---
st.sidebar.header("🎨 Sample Inputs")
st.sidebar.info("Select a sample to test the AI architecture instantly.")

# Define sample mapping (Ensure these files exist in a 'samples/' folder)
samples = {
    "None": None,
    "Sample 1: Classic Building": "C:\\p4-\\samples\\1.jpg",
    "Sample 2: Modern Complex": "C:\\p4-\\samples\\2.jpg",
    "Sample 3: Balcony Detail": "C:\\p4-\\samples\\3.jpg"
}
selected_sample = st.sidebar.selectbox("Choose a Sample Map:", list(samples.keys()))

# --- 4. Main App Logic ---
st.markdown("<h1 style='text-align:center; color:white;'>🏢AI FACADE ARCHITECT</h1>", unsafe_allow_html=True)

file = st.file_uploader("Upload Map", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Logic to determine which image to use
input_img = None
if file:
    input_img = Image.open(file).convert('RGB')
elif selected_sample != "None":
    sample_path = samples[selected_sample]
    if os.path.exists(sample_path):
        input_img = Image.open(sample_path).convert('RGB')
    else:
        st.sidebar.error(f"Sample file {sample_path} not found.")

if input_img:
    _, col1, col2, _ = st.columns([0.8, 2, 2, 0.8])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("**Source Labels**")
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
                    result = predict_facade(input_img, model)
                    st.image(result, width='stretch')
                else:
                    st.error("Model file missing.")
        else:
            st.info("Ready for synthesis")
        st.markdown('</div>', unsafe_allow_html=True)