"""
Brain Tumor Detection & Segmentation — Monolithic Streamlit App
"""
import io
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image
import nibabel as nib
import torch
import torch.nn as nn
import torchvision.models as models
import plotly.graph_objects as go
import streamlit as st

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.hero-title { font-size: 3.2rem; font-weight: 700; background: linear-gradient(135deg, #00E5FF 0%, #7B61FF 50%, #FF61DC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 0; }
.hero-sub { font-size: 1.05rem; color: #7F8EA3; margin-top: .3rem; font-weight: 400; }
.metric-card { background: linear-gradient(135deg, #111827 0%, #1a2236 100%); border: 1px solid #1E293B; border-radius: 14px; padding: 1.4rem 1.6rem; text-align: center; }
.metric-value { font-size: 2.4rem; font-weight: 700; color: #00E5FF; line-height: 1; }
.metric-label { font-size: .78rem; color: #64748B; text-transform: uppercase; letter-spacing: .1em; margin-top: .4rem; }
.badge-tumor { display:inline-block; padding:.35rem 1.2rem; border-radius:999px; background:linear-gradient(90deg,#FF4B4B,#FF8C00); color:#fff; font-weight:600; }
.badge-clear { display:inline-block; padding:.35rem 1.2rem; border-radius:999px; background:linear-gradient(90deg,#00C853,#00E5FF); color:#000; font-weight:600; }
[data-testid="stFileUploader"] { border: 2px dashed #1E3A5F; border-radius: 16px; background: #0D1526; padding: 1rem; }
.section-header { font-size:.7rem; font-weight:600; letter-spacing:.15em; text-transform:uppercase; color:#334155; border-bottom:1px solid #1E293B; padding-bottom:.4rem; margin-bottom:1rem; }
.stButton > button { background: linear-gradient(135deg, #00E5FF, #7B61FF); border: none; border-radius: 10px; color: #000; font-weight: 700; width: 100%; }
.img-caption { text-align:center; color:#475569; font-size:.8rem; margin-top:.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing & Utils ────────────────────────────────────────────────────
TARGET_SIZE = (256, 256)

def load_image(file_obj, filename: str) -> np.ndarray:
    filename = filename.lower()
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        raw = file_obj.read()
        fh = nib.FileHolder(fileobj=io.BytesIO(raw))
        img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
        data = img.get_fdata()
        if data.ndim == 4: data = data[:, :, :, 0]
        if data.ndim == 3: data = data[:, :, data.shape[2] // 2]
        return data.astype(np.float32)
    else:
        pil_img = Image.open(file_obj).convert("L")
        return np.array(pil_img, dtype=np.float32)

def normalize(image: np.ndarray) -> np.ndarray:
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8: return np.zeros_like(image, dtype=np.float32)
    return ((image - mn) / (mx - mn)).astype(np.float32)

def resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8).astype(np.float32) / 255.0

def preprocess_for_classification(image: np.ndarray) -> np.ndarray:
    img = apply_clahe(resize(normalize(image)))
    img_3ch = np.stack([img, img, img], axis=0)
    return np.expand_dims(img_3ch, axis=0).astype(np.float32)

def preprocess_for_segmentation(image: np.ndarray) -> np.ndarray:
    img = apply_clahe(resize(normalize(image)))
    return np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)

def overlay_mask(original: np.ndarray, mask: np.ndarray, alpha=0.45) -> np.ndarray:
    orig_uint8 = (resize(normalize(original)) * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
    mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    colored = base_rgb.copy()
    tumor_pixels = mask_resized > 0
    colored[tumor_pixels] = ((1 - alpha) * base_rgb[tumor_pixels] + alpha * np.array([255, 50, 50])).astype(np.uint8)
    return colored

# ── Models ───────────────────────────────────────────────────────────────────
class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(backbone.classifier[1].in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.classifier(self.feature_extractor(x).flatten(start_dim=1))

class FallbackUNet(nn.Module):
    def __init__(self, in_ch=1, f=32):
        super().__init__()
        def conv(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        def down(i, o): return nn.Sequential(nn.MaxPool2d(2), conv(i, o))
        self.inc, self.down1, self.down2, self.down3, self.down4 = conv(in_ch, f), down(f, f*2), down(f*2, f*4), down(f*4, f*8), down(f*8, f*16)
        self.up = nn.ModuleList([nn.ConvTranspose2d(f*16, f*8, 2, 2), nn.ConvTranspose2d(f*8, f*4, 2, 2), nn.ConvTranspose2d(f*4, f*2, 2, 2), nn.ConvTranspose2d(f*2, f, 2, 2)])
        self.convs = nn.ModuleList([conv(f*16, f*8), conv(f*8, f*4), conv(f*4, f*2), conv(f*2, f)])
        self.out = nn.Conv2d(f, 1, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        for i, skip in enumerate([x4, x3, x2, x1]):
            up_x = self.up[i](x5 if i==0 else x)
            dy, dx = skip.size(2) - up_x.size(2), skip.size(3) - up_x.size(3)
            up_x = nn.functional.pad(up_x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            x = self.convs[i](torch.cat([skip, up_x], dim=1))
        return self.out(x)

def build_segmenter():
    if SMP_AVAILABLE: return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1, activation=None)
    return FallbackUNet()

# ── Inference Engine & Downloading ───────────────────────────────────────────
MODEL_CONFIGS = {
    "classifier": {"filename": "classifier.pt", "gdrive_id": "YOUR_GDRIVE_FILE_ID_FOR_CLASSIFIER", "size": 50},
    "segmenter": {"filename": "segmenter.pt", "gdrive_id": "YOUR_GDRIVE_FILE_ID_FOR_SEGMENTER", "size": 110},
}

@st.cache_resource(show_spinner=False)
def load_models():
    import gdown
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models_dict = {"classifier": None, "segmenter": None}
    
    for key, cfg in MODEL_CONFIGS.items():
        dest = MODELS_DIR / cfg["filename"]
        if not dest.exists():
            if cfg["gdrive_id"].startswith("YOUR_"):
                st.warning(f"Using dummy weights for {key} (replace gdrive_id).")
                model = BrainTumorClassifier() if key == "classifier" else build_segmenter()
                torch.save({"model_state_dict": model.state_dict()}, dest)
            else:
                with st.spinner(f"⬇️ Downloading {key} (~{cfg['size']} MB)..."):
                    gdown.download(f"https://drive.google.com/uc?id={cfg['gdrive_id']}", str(dest), quiet=False)
        
        # Load logic
        state = torch.load(dest, map_location=device)
        state = state.get("model_state_dict", state)
        model = BrainTumorClassifier() if key == "classifier" else build_segmenter()
        model.load_state_dict(state)
        model.to(device).eval()
        models_dict[key] = model
        
    return models_dict["classifier"], models_dict["segmenter"], device

# ── UI Layout ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;font-size:2.5rem;">🧠</div>', unsafe_allow_html=True)
    st.markdown("## NeuroScan AI")
    st.divider()
    seg_thresh = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    cls_thresh = st.slider("Classification threshold", 0.1, 0.9, 0.5, 0.05)
    show_heatmap = st.toggle("Show probability heatmap", value=True)
    st.divider()
    st.caption("For research & educational use only. Not a medical device.")

st.markdown('<p class="hero-title">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a brain MRI scan to detect and segment tumor regions.</p><br>', unsafe_allow_html=True)

uploaded = st.file_uploader(label="Drop MRI scan here", type=["png", "jpg", "jpeg", "nii", "nii.gz"], label_visibility="collapsed")

if uploaded:
    raw = load_image(uploaded, getattr(uploaded, "name", "upload.png"))
    if st.button("🔍 Analyse Scan", use_container_width=True):
        classifier, segmenter, device = load_models()
        
        with torch.no_grad():
            cls_inp = torch.from_numpy(preprocess_for_classification(raw)).to(device)
            tumor_prob = torch.sigmoid(classifier(cls_inp)).item()
            
            seg_inp = torch.from_numpy(preprocess_for_segmentation(raw)).to(device)
            seg_logits = segmenter(seg_inp)
            raw_mask = torch.sigmoid(seg_logits).cpu().numpy().squeeze()
            
        mask = (raw_mask > seg_thresh).astype(np.uint8) * 255
        overlay = overlay_mask(raw, mask)
        tumor_det = tumor_prob >= cls_thresh
        area_pct = float(np.count_nonzero(mask)) / mask.size * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center"><span class="{"badge-tumor" if tumor_det else "badge-clear"}">{"⚠️ Tumor Detected" if tumor_det else "✅ No Tumor Detected"}</span></div><br>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        def mcard(col, val, lab): col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lab}</div></div>', unsafe_allow_html=True)
        mcard(m1, f"{tumor_prob * 100:.1f}%", "Probability"); mcard(m2, f"{area_pct:.2f}%", "Tumor Area")
        mcard(m3, f"{raw.shape[0]}×{raw.shape[1]}", "Input Res"); mcard(m4, "256×256", "Model Res")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<p class="section-header">Original</p>', unsafe_allow_html=True)
            st.image(Image.fromarray((normalize(raw)*255).astype(np.uint8), "L"), use_container_width=True)
        with col2:
            st.markdown('<p class="section-header">Mask</p>', unsafe_allow_html=True)
            st.image(Image.fromarray(mask, "L"), use_container_width=True)
        with col3:
            st.markdown('<p class="section-header">Overlay</p>', unsafe_allow_html=True)
            st.image(Image.fromarray(overlay, "RGB"), use_container_width=True)
            
        if show_heatmap:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_heat = go.Figure(go.Heatmap(z=raw_mask, colorscale=[[0,"#0A0E1A"],[0.3,"#0D3B5E"],[0.6,"#7B61FF"],[0.85,"#FF8C00"],[1,"#FF4B4B"]], showscale=True))
            fig_heat.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A", xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"))
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
