"""
NeuroScan API — Powered by Hugging Face Image Segmentation
"""
import io
import base64
import os
import requests
import numpy as np
from PIL import Image
import streamlit as st

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

# Using a standard Segformer model to ensure the Serverless API responds correctly.
# Once this works, you can hunt for a medical-specific model that supports this API.
API_URL = "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512"

# ── Helper Functions ─────────────────────────────────────────────────────────
def query_huggingface_api(image_bytes):
    """Send image to Hugging Face API and gracefully handle errors."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code} Error: {response.text}"}
    
    try:
        return response.json()
    except Exception:
        return {"error": "Failed to parse API response. The model format may not be supported by the free Serverless API."}

def overlay_mask(original_img, mask_b64, alpha=0.45):
    """Decode HF base64 mask and overlay it in red over the original image."""
    mask_data = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
    
    mask_img = mask_img.resize(original_img.size, Image.Resampling.NEAREST)
    
    orig_np = np.array(original_img.convert("RGB"))
    mask_np = np.array(mask_img)
    
    colored = orig_np.copy()
    tumor_pixels = mask_np > 128 
    
    colored[tumor_pixels] = (
        (1 - alpha) * orig_np[tumor_pixels] + 
        alpha * np.array([255, 50, 50])
    ).astype(np.uint8)
    
    return Image.fromarray(colored)

# ── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan API</p>', unsafe_allow_html=True)
st.write("Upload an MRI slice to generate a segmentation mask using Hugging Face.")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info(f"**Model Endpoint:**\n`{API_URL.split('/')[-1]}`")
    st.caption("Using Hugging Face Serverless Inference API.")
    st.divider()
    st.caption("VIBE6 INNOVATHON 2026 Submission")

# Main window
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Segment MRI via Hugging Face"):
            with st.spinner("Calling Hugging Face API..."):
                image_bytes = uploaded_file.getvalue()
                result = query_huggingface_api(image_bytes)
                
                if isinstance(result, dict) and "error" in result:
                    if "currently loading" in result["error"].lower():
                        st.warning("⏳ **Model is waking up!** Please wait 20 seconds and click the button again.")
                    else:
                        st.error(result["error"])
                
                elif isinstance(result, list) and len(result) > 0:
                    st.success("Segmentation successful!")
                    
                    best_mask_obj = result[0] 
                    confidence = best_mask_obj.get('score', 0.0)
                    mask_b64 = best_mask_obj.get('mask')
                    
                    if not mask_b64:
                        st.error("API returned a successful response, but no image mask data was found.")
                        st.stop()
                        
                    overlay_img = overlay_mask(raw_img, mask_b64)
                    mask_only_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{confidence * 100:.2f}%")
                    m2.metric("Detected Label", best_mask_obj.get('label', 'Region').title())
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    img_col1, img_col2, img_col3 = st.columns(3)
                    
                    with img_col1:
                        st.markdown('<p class="section-header">Original</p>', unsafe_allow_html=True)
                        st.image(raw_img, use_container_width=True)
                    with img_col2:
                        st.markdown('<p class="section-header">Raw Mask</p>', unsafe_allow_html=True)
                        st.image(mask_only_img, use_container_width=True)
                    with img_col3:
                        st.markdown('<p class="section-header">Overlay</p>', unsafe_allow_html=True)
                        st.image(overlay_img, use_container_width=True)
                        
                else:
                    st.info("No regions detected by the model, or invalid response format.")
