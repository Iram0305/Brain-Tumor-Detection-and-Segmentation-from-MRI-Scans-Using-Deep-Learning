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

# Using a flagship segmentation model fully supported by the new SDK
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

def draw_bounding_box(original_img, mask_pil, box_color=(255, 50, 50), thickness=4):
    """Calculates the edges of the tumor mask and draws a bounding box."""
    # Ensure mask matches original size and convert to grayscale
    mask_img = mask_pil.resize(original_img.size, Image.Resampling.NEAREST).convert("L")
    mask_np = np.array(mask_img)
    
    # Create a copy of the original image to draw on
    bounded_img = original_img.copy()
    
    # Find the coordinates where the mask is present (tumor pixels)
    tumor_indices = np.where(mask_np > 128)
    
    # If a tumor was found, calculate the bounding box coordinates
    if len(tumor_indices[0]) > 0 and len(tumor_indices[1]) > 0:
        min_y, max_y = np.min(tumor_indices[0]), np.max(tumor_indices[0])
        min_x, max_x = np.min(tumor_indices[1]), np.max(tumor_indices[1])
        
        # Draw the rectangle
        draw = ImageDraw.Draw(bounded_img)
        draw.rectangle([min_x, min_y, max_x, max_y], outline=box_color, width=thickness)
        
    return bounded_img

# ── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan API</p>', unsafe_allow_html=True)
st.write("Upload an MRI slice to detect anomalies using the Hugging Face SDK.")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info(f"**Model Endpoint:**\n`{MODEL_ID}`")
    st.caption("Using official `huggingface_hub` InferenceClient.")
    st.divider()
    st.caption("VIBE6 INNOVATHON 2026 Submission")

# Main window
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convert upload to a standard PIL Image for the UI and saving
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Run Detection"):
            
            if HF_TOKEN == "YOUR_HUGGINGFACE_API_TOKEN_HERE":
                st.error("⚠️ **Missing Token:** Please set your Hugging Face token in the code or Streamlit Secrets!")
                st.stop()
                
            with st.spinner("Analyzing via Hugging Face SDK..."):
                
                # Create a secure temporary file to bypass the Content-Type error
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    raw_img.save(tmp_file, format="JPEG")
                    temp_path = tmp_file.name
                
                try:
                    result = query_huggingface_sdk(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Handle API Errors
                if isinstance(result, dict) and "error" in result:
                    st.error(f"API Error: {result['error']}")
                
                # Handle Success
                elif isinstance(result, list) and len(result) > 0:
                    st.success("Detection successful!")
                    
                    best_mask_obj = result[0] 
                    confidence = best_mask_obj.get('score', 0.0)
                    mask_pil = best_mask_obj.get('mask')
                    
                    # Generate the bounding box image
                    bounded_img = draw_bounding_box(raw_img, mask_pil)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{confidence * 100:.2f}%")
                    m2.metric("Detected Label", best_mask_obj.get('label', 'Region').title())
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Clean 2-column layout for the images
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown('<p class="section-header">Original</p>', unsafe_allow_html=True)
                        st.image(raw_img, use_container_width=True)
                    with img_col2:
                        st.markdown('<p class="section-header">Detection Box</p>', unsafe_allow_html=True)
                        st.image(bounded_img, use_container_width=True)
                        
                else:
                    st.info("No anomalies detected by the model.")
