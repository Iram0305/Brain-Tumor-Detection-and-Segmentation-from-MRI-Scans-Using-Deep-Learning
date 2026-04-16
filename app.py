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
# Best practice: Store your token in Streamlit secrets (.streamlit/secrets.toml)
# st.secrets["HF_TOKEN"]
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_API_TOKEN_HERE")

# You can swap this URL with any image-segmentation model on Hugging Face 
# that supports the Serverless Inference API (e.g., yaraa11/brain-tumor-segmentation)
API_URL = "https://api-inference.huggingface.co/models/yaraa11/brain-tumor-segmentation"

# ── Helper Functions ─────────────────────────────────────────────────────────
def query_huggingface_api(image_bytes):
    """Send image to Hugging Face API and get segmentation mask."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

def overlay_mask(original_img, mask_b64, alpha=0.45):
    """Decode HF base64 mask and overlay it in red over the original image."""
    # Decode the mask
    mask_data = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
    
    # Ensure mask dimensions perfectly match the original image
    mask_img = mask_img.resize(original_img.size, Image.Resampling.NEAREST)
    
    # Convert to numpy arrays
    orig_np = np.array(original_img.convert("RGB"))
    mask_np = np.array(mask_img)
    
    # Create the red overlay
    colored = orig_np.copy()
    tumor_pixels = mask_np > 128  # Threshold the mask
    
    colored[tumor_pixels] = (
        (1 - alpha) * orig_np[tumor_pixels] + 
        alpha * np.array([255, 50, 50]) # Red color
    ).astype(np.uint8)
    
    return Image.fromarray(colored)

# ── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan API</p>', unsafe_allow_html=True)
st.write("Upload an MRI slice to generate a precise segmentation mask using Hugging Face.")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info(f"**Model Endpoint:**\n`{API_URL.split('/')[-1]}`")
    st.caption("Using Hugging Face Serverless Inference API.")

# Main window
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display preview
    raw_img = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Segment MRI via Hugging Face"):
            with st.spinner("Calling Hugging Face API..."):
                image_bytes = uploaded_file.getvalue()
                result = query_huggingface_api(image_bytes)
                
                # Handle API waking up or errors
                if isinstance(result, dict) and "error" in result:
                    if "currently loading" in result["error"].lower():
                        st.warning(f"⏳ **Model is waking up!** Estimated time: {result.get('estimated_time', 20):.1f} seconds. Please wait and click the button again.")
                    else:
                        st.error(f"API Error: {result['error']}")
                
                # Handle successful segmentation
                elif isinstance(result, list) and len(result) > 0:
                    st.success("Segmentation successful!")
                    
                    # Hugging Face image-segmentation APIs return a list of dictionaries
                    # Usually, the first object is our mask. (Depending on the model, you might need to filter by 'label')
                    best_mask_obj = result[0] 
                    confidence = best_mask_obj.get('score', 0.0)
                    mask_b64 = best_mask_obj.get('mask')
                    
                    # Generate the overlay image
                    overlay_img = overlay_mask(raw_img, mask_b64)
                    mask_only_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                    
                    # Display metrics
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{confidence * 100:.2f}%")
                    m2.metric("Detected Label", best_mask_obj.get('label', 'Tumor').title())
                    
                    # Display Images
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
                    st.info("No tumor regions detected by the model, or invalid response format.")
