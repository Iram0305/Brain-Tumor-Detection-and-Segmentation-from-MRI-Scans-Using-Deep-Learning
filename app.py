"""
NeuroScan API — Premium VIBE6 Dashboard
Powered by Hugging Face InferenceClient
"""
import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from huggingface_hub import InferenceClient

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI | VIBE6", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── Premium Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0B0F19; color: #E2E8F0; }
.hero-title { font-size: 3rem; font-weight: 700; letter-spacing: -0.02em; background: linear-gradient(135deg, #00E5FF, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; padding-bottom: 0; }
.hero-subtitle { font-size: 1.1rem; color: #94A3B8; font-weight: 400; margin-top: 0.2rem; margin-bottom: 2rem; }
.stButton > button { background: linear-gradient(135deg, #3B82F6, #2563EB); border: 1px solid #60A5FA; border-radius: 8px; color: #FFF; font-weight: 600; width: 100%; padding: 0.6rem; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
.metric-container { background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.5rem; text-align: center; }
.metric-value { font-size: 2.5rem; font-weight: 700; color: #00E5FF; }
.metric-label { font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.section-header { font-size: 0.9rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #94A3B8; border-bottom: 1px solid #1E293B; padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 1.5rem; }
[data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
[data-testid="stFileUploader"] { background-color: #111827; border: 2px dashed #334155; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── API Configuration ────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_API_TOKEN_HERE")
MODEL_ID = "facebook/detr-resnet-50-panoptic"

# ── Helper Functions ─────────────────────────────────────────────────────────
def query_huggingface_sdk(file_path):
    """Segment image using Hugging Face SDK."""
    client = InferenceClient(token=HF_TOKEN)
    try:
        return client.image_segmentation(file_path, model=MODEL_ID)
    except Exception as e:
        return {"error": str(e)}

def draw_bounding_box(original_img, mask_pil, box_color=(0, 229, 255), thickness=3):
    """Draws a clean bounding box, filtering out full-image background errors."""
    mask_img = mask_pil.resize(original_img.size, Image.Resampling.NEAREST).convert("L")
    mask_np = np.array(mask_img)
    
    bounded_img = original_img.copy()
    tumor_indices = np.where(mask_np > 128)
    
    if len(tumor_indices[0]) > 0 and len(tumor_indices[1]) > 0:
        min_y, max_y = np.min(tumor_indices[0]), np.max(tumor_indices[0])
        min_x, max_x = np.min(tumor_indices[1]), np.max(tumor_indices[1])
        
        # LOGIC FIX: Filter out masks that cover more than 90% of the image (backgrounds)
        mask_area = (max_y - min_y) * (max_x - min_x)
        total_area = mask_np.shape[0] * mask_np.shape[1]
        
        if mask_area < (total_area * 0.90):
            draw = ImageDraw.Draw(bounded_img)
            # Add a 10px padding for aesthetics
            pad = 10
            box_coords = [
                max(0, min_x - pad), max(0, min_y - pad),
                min(original_img.width, max_x + pad), min(original_img.height, max_y + pad)
            ]
            draw.rectangle(box_coords, outline=box_color, width=thickness)
            
    return bounded_img

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Config")
    st.info(f"**Endpoint:**\n`{MODEL_ID}`")
    st.caption("Active connection: Hugging Face Serverless API.")
    
    st.divider()
    
    st.markdown("### 🏆 VIBE6 INNOVATHON 2026")
    st.caption("SVKM's NMIMS, Indore")
    st.markdown("""
    **Project Team:**
    * Iram Shaikh
    * Aayush Nayak
    * Ritwik Shetty
    * Rish Patil
    """)

# ── Main UI Layout ───────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-precision anomaly detection and localization architecture.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Initialize Pipeline: Upload MRI Scan", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button("🔍 Execute Detection Protocol"):
            
            if HF_TOKEN == "YOUR_HUGGINGFACE_API_TOKEN_HERE":
                st.error("⚠️ **System Alert:** Hugging Face API token is missing or invalid.")
                st.stop()
                
            with st.spinner("Transmitting to LPU / GPU architecture..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    raw_img.save(tmp_file, format="JPEG")
                    temp_path = tmp_file.name
                
                try:
                    result = query_huggingface_sdk(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                if isinstance(result, dict) and "error" in result:
                    st.error(f"Endpoint Error: {result['error']}")
                
                elif isinstance(result, list) and len(result) > 0:
                    best_mask_obj = result[0] 
                    confidence = best_mask_obj.get('score', 0.0)
                    mask_pil = best_mask_obj.get('mask')
                    
                    bounded_img = draw_bounding_box(raw_img, mask_pil)
                    
                    st.success("Target acquired. Analysis complete.")
                    
                    # Premium Metric Cards
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{confidence * 100:.2f}%</div><div class="metric-label">Confidence Score</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-container"><div class="metric-value" style="color:#FFF;">{best_mask_obj.get("label", "Anomaly").title()}</div><div class="metric-label">Identified Classification</div></div>', unsafe_allow_html=True)
                    
                    # Image Comparisons
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.markdown('<p class="section-header">Source Radiograph</p>', unsafe_allow_html=True)
                        st.image(raw_img, use_container_width=True)
                    with img_col2:
                        st.markdown('<p class="section-header">Bounding Box Localization</p>', unsafe_allow_html=True)
                        st.image(bounded_img, use_container_width=True)
                        
                else:
                    st.info("Analysis nominal. No localized anomalies detected.")
