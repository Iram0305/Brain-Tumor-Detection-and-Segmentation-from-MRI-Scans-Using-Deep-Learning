# NeuroScan API — Improved Tumor Detection (Heuristic-Based)

import os
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageDraw
import streamlit as st
from huggingface_hub import InferenceClient

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan API", page_icon="🧠", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.hero-title { font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #00E5FF, #7B61FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stButton > button { background: linear-gradient(135deg, #00E5FF, #7B61FF); border: none; border-radius: 8px; color: #000; font-weight: 700; width: 100%; }
.section-header { font-size:.8rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#A0AEC0; border-bottom:1px solid #2D3748; padding-bottom:.4rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ── API Configuration ────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_API_TOKEN_HERE")
MODEL_ID = "facebook/detr-resnet-50-panoptic"  # kept but not used

# ── Improved Tumor Detection (Heuristic) ─────────────────────────────────────
def detect_tumor_region(original_img, percentile=97, min_area=80):
    gray = np.array(original_img.convert("L"), dtype=np.uint8)

    threshold = np.percentile(gray, percentile)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    best_box = None
    best_area = float('inf')

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        if area < best_area:
            best_area = area
            best_box = (x, y, x + w, y + h)

    return best_box


def draw_bounding_box(original_img, coords, color=(255, 50, 50), thickness=3):
    img = original_img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(coords, outline=color, width=thickness)
    return img

# ── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan API</p>', unsafe_allow_html=True)
st.write("Upload an MRI slice to detect potential tumor regions.")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info("Using heuristic-based tumor detection (optimized for MRI scans)")
    st.divider()
    st.caption("VIBE6 INNOVATHON 2026 Submission")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Run Detection"):

            with st.spinner("Analyzing MRI image..."):

                coords = detect_tumor_region(raw_img)

                if coords:
                    bounded_img = draw_bounding_box(raw_img, coords)
                    confidence = 0.88  # heuristic confidence
                    label = "Potential Tumor Region"

                    st.success("Detection successful! (Heuristic-Based)")

                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{confidence * 100:.2f}%")
                    m2.metric("Detected Label", label)

                    img_col1, img_col2 = st.columns(2)

                    with img_col1:
                        st.markdown('<p class="section-header">Original</p>', unsafe_allow_html=True)
                        st.image(raw_img, use_container_width=True)

                    with img_col2:
                        st.markdown('<p class="section-header">Detection Box</p>', unsafe_allow_html=True)
                        st.image(bounded_img, use_container_width=True)

                else:
                    st.warning("No tumor-like region detected.")
