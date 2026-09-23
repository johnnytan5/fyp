"""Streamlit UI for crack segmentation inference."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from infer import load_model, load_sahi_model, run_inference

st.set_page_config(page_title="Crack Segment", layout="wide")
st.title("🧱 Crack Segmentation Inference")
st.caption("YOLOv8s-Seg trained on CrackSeg9k + OmniCrack30k")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_sahi_model(conf: float):
    return load_sahi_model(conf)


# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    use_sahi = st.checkbox("Enable SAHI sliced inference", value=False,
                           help="Split image into tiles for better detection on large/dense images")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Lower = more detections (noisier), Higher = fewer detections (cleaner)")
    slice_size = 512
    if use_sahi:
        slice_size = st.select_slider(
            "SAHI slice size", options=[256, 320, 384, 512, 640], value=512,
            help="Smaller = each tile upscaled to 640 = thin cracks magnified (better recall, slower)")
        st.info(f"Tiles at {slice_size}×{slice_size}, 20% overlap, fed to model at 640 "
                f"(~{640 / slice_size:.1f}× magnify)")

# --- Main area ---
uploaded = st.file_uploader("Upload a crack/drone image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        if st.button("Run Inference", type="primary"):
            with st.spinner("Running inference..."):
                model = get_sahi_model(conf) if use_sahi else get_model()
                annotated, count = run_inference(image_np, model, use_sahi=use_sahi,
                                                 conf=conf, slice_size=slice_size)
                st.image(annotated, use_container_width=True)
                st.success(f"Found {count} crack{'s' if count != 1 else ''}")
else:
    st.info("Upload an image to get started.")
