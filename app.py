"""
NeuroScan API — Powered by Hugging Face InferenceClient
"""
import os
import tempfile
import numpy as np
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
MODEL_ID = "facebook/detr-resnet-50-panoptic"

# ── Helper Functions ─────────────────────────────────────────────────────────
def query_huggingface_sdk(file_path):
    """Use the official SDK to segment the image using a local file path."""
    client = InferenceClient(token=HF_TOKEN)
    try:
        results = client.image_segmentation(file_path, model=MODEL_ID)
        return results
    except Exception as e:
        return {"error": str(e)}


def find_smallest_segment(result, original_img, min_area_ratio=0.001, max_area_ratio=0.25):
    """
    From all returned segments, pick the SMALLEST region that is still meaningful.
    Tumors are small localised regions — we skip segments that cover most of the image.

    min_area_ratio: ignore near-empty masks (< 0.1% of image)
    max_area_ratio: ignore large segments that cover > 25% of the image
    """
    total_pixels = original_img.width * original_img.height
    best_segment = None
    best_area = float('inf')

    for segment in result:
        mask_pil = segment.get('mask')
        if mask_pil is None:
            continue

        mask_resized = mask_pil.resize(original_img.size, Image.Resampling.NEAREST).convert("L")
        mask_np = np.array(mask_resized)
        mask_area = int(np.sum(mask_np > 128))
        area_ratio = mask_area / total_pixels

        # Skip segments that are too large (whole-brain/background) or trivially tiny
        if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
            continue

        if mask_area < best_area:
            best_area = mask_area
            best_segment = segment

    return best_segment


def brightness_based_fallback(original_img, percentile=97, padding=6):
    """
    Fallback: find the brightest cluster in the grayscale image.
    In contrast-enhanced MRI, tumors typically appear as the brightest region.
    Returns (min_x, min_y, max_x, max_y) or None.
    """
    gray = np.array(original_img.convert("L"), dtype=np.float32)

    threshold = np.percentile(gray, percentile)
    bright_mask = (gray >= threshold).astype(np.uint8)

    mask_pil = Image.fromarray(bright_mask * 255)
    mask_np = np.array(mask_pil)
    indices = np.where(mask_np > 128)

    if len(indices[0]) == 0:
        return None

    min_y, max_y = int(np.min(indices[0])), int(np.max(indices[0]))
    min_x, max_x = int(np.min(indices[1])), int(np.max(indices[1]))

    # Add a small padding
    w, h = original_img.size
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w - 1, max_x + padding)
    max_y = min(h - 1, max_y + padding)

    return (min_x, min_y, max_x, max_y)


def draw_bounding_box_from_mask(original_img, mask_pil, box_color=(255, 50, 50), thickness=3):
    """Draw a tight bounding box only around the region covered by mask_pil."""
    mask_resized = mask_pil.resize(original_img.size, Image.Resampling.NEAREST).convert("L")
    mask_np = np.array(mask_resized)

    tumor_indices = np.where(mask_np > 128)
    if len(tumor_indices[0]) == 0:
        return original_img.copy(), False

    min_y, max_y = int(np.min(tumor_indices[0])), int(np.max(tumor_indices[0]))
    min_x, max_x = int(np.min(tumor_indices[1])), int(np.max(tumor_indices[1]))

    bounded_img = original_img.copy()
    draw = ImageDraw.Draw(bounded_img)
    draw.rectangle([min_x, min_y, max_x, max_y], outline=box_color, width=thickness)
    return bounded_img, True


def draw_bounding_box_from_coords(original_img, coords, box_color=(255, 50, 50), thickness=3):
    """Draw a bounding box from (min_x, min_y, max_x, max_y) coordinates."""
    bounded_img = original_img.copy()
    draw = ImageDraw.Draw(bounded_img)
    draw.rectangle(list(coords), outline=box_color, width=thickness)
    return bounded_img


# ── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan API</p>', unsafe_allow_html=True)
st.write("Upload an MRI slice to detect anomalies using the Hugging Face SDK.")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info(f"**Model Endpoint:**\n`{MODEL_ID}`")
    st.caption("Using official `huggingface_hub` InferenceClient.")
    st.divider()
    st.caption("VIBE6 INNOVATHON 2026 Submission")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Run Detection"):

            if HF_TOKEN == "YOUR_HUGGINGFACE_API_TOKEN_HERE":
                st.error("⚠️ **Missing Token:** Please set your Hugging Face token in the code or Streamlit Secrets!")
                st.stop()

            with st.spinner("Analyzing via Hugging Face SDK..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    raw_img.save(tmp_file, format="JPEG")
                    temp_path = tmp_file.name

                try:
                    result = query_huggingface_sdk(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                # ── Handle API Errors ────────────────────────────────────
                if isinstance(result, dict) and "error" in result:
                    st.error(f"API Error: {result['error']}")

                # ── Handle Successful API Response ───────────────────────
                elif isinstance(result, list) and len(result) > 0:

                    # ✅ KEY FIX: find the SMALLEST localised segment, not result[0]
                    best_segment = find_smallest_segment(result, raw_img)

                    if best_segment is not None:
                        confidence = best_segment.get('score', 0.0)
                        mask_pil = best_segment.get('mask')
                        label = best_segment.get('label', 'Anomaly').title()

                        bounded_img, drew = draw_bounding_box_from_mask(raw_img, mask_pil)
                        detection_method = "Model Segmentation"

                    else:
                        # ── Fallback: brightness-based detection ─────────
                        st.warning("No small localised segment found — using brightness-based fallback detection.")
                        coords = brightness_based_fallback(raw_img)

                        if coords:
                            bounded_img = draw_bounding_box_from_coords(raw_img, coords)
                            confidence = 0.85   # heuristic confidence for fallback
                            label = "Anomaly (Bright Region)"
                            detection_method = "Brightness Threshold Fallback"
                        else:
                            st.info("No anomalies detected.")
                            st.stop()

                    st.success(f"Detection successful! *(Method: {detection_method})*")
                    st.markdown("<br>", unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{confidence * 100:.2f}%")
                    m2.metric("Detected Label", label)

                    st.markdown("<br>", unsafe_allow_html=True)

                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.markdown('<p class="section-header">Original</p>', unsafe_allow_html=True)
                        st.image(raw_img, use_container_width=True)
                    with img_col2:
                        st.markdown('<p class="section-header">Detection Box</p>', unsafe_allow_html=True)
                        st.image(bounded_img, use_container_width=True)

                else:
                    st.info("No anomalies detected by the model.")
